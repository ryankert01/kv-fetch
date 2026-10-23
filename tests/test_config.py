from kvfetch.config import load_config


def test_inherit_merges_and_replace_marker(tmp_path):
    (tmp_path / "parent.yaml").write_text("a: {x: 1, y: 2}\nb: {p: 1, q: 2}\n")
    (tmp_path / "child.yaml").write_text("inherit: parent.yaml\na: {y: 3}\nb: {_replace: true, r: 9}\n")
    cfg = load_config(tmp_path / "child.yaml", ["a.z=5"])
    assert cfg["a"] == {"x": 1, "y": 3, "z": 5}       # merged, then overridden
    assert cfg["b"] == {"r": 9}                        # replaced, marker dropped
