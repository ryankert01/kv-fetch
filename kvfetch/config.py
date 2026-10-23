"""YAML config loading.

A config may name a parent with ``inherit: <path relative to this file>``; the child
is deep-merged over the parent. ``--set a.b.c=value`` overrides are applied last and
parsed as YAML scalars, so ``--set definitions.w=16`` yields an int.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is not None:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_path(cfg, key, yaml.safe_load(raw))
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def get_path(cfg: dict, key: str, default: Any = KeyError) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is KeyError:
                raise KeyError(f"config key '{key}' missing (at '{part}')")
            return default
        node = node[part]
    return node


def set_path(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", help="path to a YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value (repeatable)")


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.config, args.overrides)
