"""K0: trace-driven simulator of vLLM's hybrid prefix cache (EXPERIMENTS.md, "Prefix caching for hybrid models").

A cached prefix of a hybrid model needs the attention KV of every block before the hit and one GDN
checkpoint (all GDN layers' states) exactly at the hit. Requests are replayed in arrival order, one
instance per shard; each instance has an LRU GPU cache and optionally an LRU host pool fed write-through
(as vLLM's OffloadingConnector). Cost of a request = prefill compute of the tokens after its hit
(fitted per-token cost) + host -> GPU copies of what the hit used from the host pool.

Trace blocks: Mooncake hash ids are content-chained (verified: an id always has the same position and
predecessor), so one id names its whole prefix. The traces hash prompts only; a follow-up whose prompt
extends an earlier prompt's full blocks and is at least that request's prompt + output long is assumed
to carry that output: its next blocks are aliased to the earlier request's own output blocks.

Placements (which checkpoints a request keeps; positions are block boundaries):
  retention0  vLLM main default: the prompt's replay boundary ((prompt - 1) // block) and the junction
  dense       retention None: every prefill chunk end, the replay boundary, the junction, every decode block
  periodic    retention k: replay boundary, junction, and every multiple of k (prefill and decode)
  online      replay boundary, junction, and the last full block of prompt + output
  oracle      exactly the boundaries a later request in the same instance will hit (future known)
The junction is where a request's KV hit ran past its checkpoint hit; host "vllm" finds it only in the
GPU cache and, as vLLM's default offload_prompt_only, stores only prompt blocks and states; host "fixed"
finds junctions in both tiers and stores everything.

v2 (after the audit of 2026-10-28): an output block is aliased only to the first follow-up that claims it,
output full blocks are (prompt + output - 1) // block, host "vllm" is prompt-only, and periodic retention
keeps prefill states only at chunk ends.

Usage: k0_simulate.py CONFIG OUT.json [--set k=v ...]
"""
import argparse
import hashlib
import json
import time
import zlib
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kvfetch.config import load_config
from kvfetch.provenance import REPO_ROOT, provenance

GiB = 1 << 30


def load_trace(spec: dict) -> list[dict]:
    path = REPO_ROOT / spec["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == spec["sha256"], f"{path}: sha256 {digest} != config"
    reqs = [json.loads(line) for line in open(path)]
    assert all(a["timestamp"] <= b["timestamp"] for a, b in zip(reqs, reqs[1:])), "trace not in arrival order"
    return reqs


def build_chains(reqs: list[dict], B: int) -> tuple[list[dict], dict]:
    """Per request: prompt length, output length, and its sequence chain (prompt full blocks, then the
    blocks completed by its output; ids of output blocks are new negative ints)."""
    alias: dict[int, int] = {}
    claimed: set[int] = set()              # output blocks already carried by an earlier follow-up
    prompt_end: dict[int, int] = {}        # chain id of a prompt's last full block -> request index
    out, next_id, n_alias = [], -1, 0
    for i, r in enumerate(reqs):
        L, O, h = r["input_length"], r["output_length"], r["hash_ids"]
        nf = L // B
        c = [alias.get(x, x) for x in h[:nf]]
        for j in range(nf - 1, -1, -1):    # deepest earlier prompt this one extends
            a = prompt_end.get(c[j])
            if a is not None:
                A = out[a]
                if L >= A["L"] + A["O"]:      # only the first follow-up claims an output block
                    for k in range(A["nf"], min(len(A["seq"]), nf)):
                        if h[k] in alias:
                            continue
                        if A["seq"][k] in claimed:
                            break
                        alias[h[k]] = A["seq"][k]
                        claimed.add(A["seq"][k])
                        n_alias += 1
                        c[k] = A["seq"][k]
                break
        seq = list(c)
        for _ in range(nf, (L + O - 1) // B):   # the last sampled token's KV is never written
            seq.append(next_id)
            next_id -= 1
        out.append({"L": L, "O": O, "nf": nf, "seq": seq})
        if nf:
            prompt_end[c[nf - 1]] = i
    return out, {"aliased_blocks": n_alias, "output_blocks": -next_id - 1}


def shard_of(reqs: list[dict], n_inst: int, max_share: float) -> list[int]:
    counts = Counter(x for r in reqs for x in r["hash_ids"])
    cap = max_share * len(reqs)
    shards = []
    for r in reqs:
        key = next((x for x in r["hash_ids"] if counts[x] <= cap), r["hash_ids"][-1])
        shards.append(zlib.crc32(str(key).encode()) % n_inst)
    return shards


def oracle_needs(chains: list[dict], B: int) -> list[dict]:
    """For each request, the checkpoint keys later requests (same list) hit at, with their last use index.
    A request's oracle hit is the longest prefix of its prompt chain present in an earlier sequence."""
    seen: set[int] = set()
    last_use: dict[int, int] = {}
    for i, q in enumerate(chains):
        maxb = (q["L"] - 1) // B
        j = 0
        while j < maxb and q["seq"][j] in seen:
            j += 1
        if j:
            last_use[q["seq"][j - 1]] = i
        seen.update(q["seq"])
    return last_use


class Tier:
    def __init__(self, cap: int):
        self.cap, self.used, self.d = cap, 0, OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, k, size: int):
        if k in self.d:
            self.d.move_to_end(k)
            return
        self.d[k] = size
        self.used += size
        while self.used > self.cap:
            _, s = self.d.popitem(last=False)
            self.used -= s


def simulate(chains: list[dict], pol: dict, gpu_bytes: int, host_bytes: int, k0: dict) -> dict:
    B = k0["block_tokens"]
    kvb = B * k0["bytes"]["kv_per_token"]
    ckb = k0["bytes"]["checkpoint"][pol["precision"]]
    a, b, bw = k0["cost"]["a_ms"], k0["cost"]["b_ms"], k0["cost"]["h2d_GBps"] * 1e6   # bytes per ms
    chunk_blocks = k0["prefill_chunk_tokens"] // B
    place, host_mode = pol["placement"], pol["host"]
    per_blocks = pol.get("period_tokens", 0) // B
    gpu = Tier(gpu_bytes)
    host = Tier(host_bytes) if host_mode != "none" else None
    needs = oracle_needs(chains, B) if place == "oracle" else {}

    def avail(k):
        return k in gpu or (host is not None and k in host)

    tot = Counter()
    for i, q in enumerate(chains):
        L, O, nf, seq = q["L"], q["O"], q["nf"], q["seq"]
        maxb = (L - 1) // B
        K = 0                                   # longest KV prefix (blocks) in either tier
        while K < maxb and avail(("kv", seq[K])):
            K += 1
        j = K                                   # deepest checkpoint at or below it
        while j > 0 and not avail(("ck", seq[j - 1])):
            j -= 1
        if host_mode == "fixed" or host is None:
            junction = K
        else:                                   # vLLM: junction from the local (GPU) lookup only
            junction = 0
            while junction < maxb and ("kv", seq[junction]) in gpu:
                junction += 1
        junction = junction if junction > j else 0
        reload = 0
        for k in [("kv", x) for x in seq[:j]] + ([("ck", seq[j - 1])] if j else []):
            if k not in gpu:
                reload += ckb if k[0] == "ck" else kvb
        s, n = j * B, L - j * B
        tot["compute_ms"] += n * a + b * n * (s + (n - 1) / 2)
        tot["reload_ms"] += reload / bw
        tot["recomputed_tokens"] += n
        tot["prompt_tokens"] += L
        tot["hit_tokens"] += s
        tot["requests"] += 1
        tot["kv_hit_tokens"] += K * B

        m = len(seq)                            # boundaries reachable: (j, m]
        replay = maxb
        prefill_ends = set(range(j + chunk_blocks, replay, chunk_blocks))   # chunk ends below the replay boundary
        decode_bounds = range(-(-L // B), m + 1)
        keep = set()
        if place in ("retention0", "dense", "periodic", "online"):
            if replay > j:
                keep.add(replay)
            if junction > j:
                keep.add(junction)
        if place == "dense":
            keep.update(prefill_ends)
            keep.update(decode_bounds)
        elif place == "periodic":                # prefill states exist only at chunk ends
            keep.update(p for p in prefill_ends if p % per_blocks == 0)
            keep.update(p for p in decode_bounds if p % per_blocks == 0)
        elif place == "online":
            if m > replay:
                keep.add(m)
        elif place == "oracle":
            keep.update(p for p in range(j + 1, m + 1) if needs.get(seq[p - 1], -1) > i)
        keep = {p for p in keep if 0 < p <= m}
        tot["checkpoints_kept"] += len(keep)
        touched = keep | ({j} if j else set())  # the hit's checkpoint, and the ones this request stores
        tiers = (gpu, host) if host is not None else (gpu,)
        host_top = L // B if host_mode == "vllm" else m   # vLLM default offload_prompt_only: prompt blocks only
        for p in range(m, 0, -1):               # vLLM frees a request's blocks tail first
            for tier in tiers:                  # host: write-through, as the OffloadingConnector stores
                if tier is host and p > host_top:
                    continue
                if p in touched:
                    tier.touch(("ck", seq[p - 1]), ckb)
                tier.touch(("kv", seq[p - 1]), kvb)
    tot["gpu_kv_tokens_end"] = sum(1 for k in gpu.d if k[0] == "kv") * B
    tot["gpu_ck_bytes_end"] = sum(s for k, s in gpu.d.items() if k[0] == "ck")
    tot["gpu_used_end"] = gpu.used
    return dict(tot)


def run_one(args):
    chains, pol, gpu_bytes, host_bytes, k0 = args
    return simulate(chains, pol, gpu_bytes, host_bytes, k0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    k0 = cfg["k0"]
    B = k0["block_tokens"]
    t0 = time.time()

    pols = [dict(p) for p in k0["existing"]] + [dict(k0["ours"], ours=True)] + [dict(p) for p in k0["ablations"]]
    expanded = []
    for p in pols:
        if p["placement"] == "periodic":
            expanded += [dict(p, period_tokens=t) for t in k0["periodic_tokens"]]
        else:
            expanded.append(p)

    def name(p):
        per = f"@{p['period_tokens']}" if "period_tokens" in p else ""
        return f"{p['placement']}{per}/{p['precision']}/host-{p['host']}"

    results = {}
    trace_info = {}
    jobs, keys = [], []
    for tname, spec in k0["traces"].items():
        reqs = load_trace(spec)
        chains, alias_info = build_chains(reqs, B)
        shards = shard_of(reqs, k0["instances"], k0["session_key_max_share"])
        per_shard = [[c for c, s in zip(chains, shards) if s == k] for k in range(k0["instances"])]
        trace_info[tname] = {"requests": len(reqs), **alias_info,
                             "requests_per_instance": [len(x) for x in per_shard]}
        for g in k0["gpu_cache_GiB"]:
            for p in expanded:
                host = g * k0["host_per_gpu"] * GiB if p["host"] != "none" else 0
                for k, sh in enumerate(per_shard):
                    jobs.append((sh, p, g * GiB, host, k0))
                    keys.append((tname, g, name(p), k))
    with ProcessPoolExecutor(k0["workers"]) as ex:
        outs = list(ex.map(run_one, jobs, chunksize=1))
    for (tname, g, pname, k), o in zip(keys, outs):
        r = results.setdefault(tname, {}).setdefault(str(g), {}).setdefault(pname, Counter())
        r.update(o)
    fixed = k0["cost"]["fixed_ms"]
    summary, gate = {}, {}
    for tname, by_g in results.items():
        for g, by_p in by_g.items():
            rows = {}
            for pname, t in by_p.items():
                t = dict(t)
                t["prefill_ms"] = t["compute_ms"] + t["reload_ms"]
                t["prefill_ms_with_fixed"] = t["prefill_ms"] + fixed * t["requests"]
                t["hit_rate"] = t["hit_tokens"] / t["prompt_tokens"]
                rows[pname] = t
            existing = {name(p): rows[name(p)]["prefill_ms"] for p in expanded
                        if any(p["placement"] == e["placement"] and p["precision"] == e["precision"]
                               and p["host"] == e["host"] for e in k0["existing"])}
            best = min(existing, key=existing.get)
            ours = rows[name(k0["ours"])]
            cut = 1 - ours["prefill_ms"] / existing[best]
            cut_fixed = 1 - ours["prefill_ms_with_fixed"] / rows[best]["prefill_ms_with_fixed"]
            summary.setdefault(tname, {})[g] = {"policies": rows, "best_existing": best, "cut": cut,
                                                "cut_with_fixed": cut_fixed}
            gate.setdefault(g, {})[tname] = cut
            print(f"{tname:12s} {g:>3s} GiB  best existing {best:32s} cut {cut:+.3f} (with fixed {cut_fixed:+.3f})", flush=True)
    passed_sizes = [g for g, by_t in gate.items() if all(c >= k0["gate"]["min_cut"] for c in by_t.values())]
    prov = provenance(cfg, model=cfg["hybrid"]["models"][k0["model"]]["name"], context_length=None,
                      script=str(Path(__file__).resolve()), settings=k0, traces=trace_info,
                      seconds=round(time.time() - t0, 1))
    json.dump({"provenance": prov, "summary": summary,
               "gate": {"cut_by_size": gate, "passed_sizes": passed_sizes, "passed": bool(passed_sizes)}},
              open(args.out, "w"), indent=1)
    print("gate:", "PASSED at " + ", ".join(passed_sizes) + " GiB" if passed_sizes else "FAILED", flush=True)


if __name__ == "__main__":
    main()
