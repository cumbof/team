from pathlib import Path

import pytest

from team.workspace import SharedWorkspace, list_dir_files, parse_file_blocks


def test_parse_file_blocks_basic() -> None:
    text = (
        "intro\n"
        "```file:foo/bar.md\n"
        "# Title\n"
        "body\n"
        "```\n"
        "more\n"
        "```file:src/x.py\n"
        "print(1)\n"
        "```\n"
    )
    blocks = parse_file_blocks(text)
    assert [p for p, _ in blocks] == ["foo/bar.md", "src/x.py"]
    assert "print(1)" in dict(blocks)["src/x.py"]


def test_workspace_apply_reply_writes(tmp_path: Path) -> None:
    ws = SharedWorkspace(tmp_path)
    text = "ok\n```file:notes/n.md\nhello\n```\n"
    writes = ws.apply_reply(text)
    assert len(writes) == 1
    assert writes[0].path == "notes/n.md"
    assert (ws.shared / "notes/n.md").read_text() == "hello\n"
    assert "notes/n.md" in ws.list_files()
    assert "notes/n.md" in ws.recent_changes()


def test_workspace_rejects_traversal(tmp_path: Path) -> None:
    ws = SharedWorkspace(tmp_path)
    text = "```file:../escape.txt\nbad\n```"
    writes = ws.apply_reply(text)
    assert writes == []
    assert not (tmp_path / "escape.txt").exists()


def test_workspace_overwrite(tmp_path: Path) -> None:
    ws = SharedWorkspace(tmp_path)
    ws.apply_reply("```file:a.txt\nv1\n```")
    res = ws.apply_reply("```file:a.txt\nv2\n```")
    assert res[0].created is False
    assert (ws.shared / "a.txt").read_text() == "v2\n"


def test_workspace_touch_updates_recent(tmp_path: Path) -> None:
    ws = SharedWorkspace(tmp_path)
    # touch a path that was not written in this session
    ws.touch("existing/file.md")
    assert "existing/file.md" in ws.recent_changes()


# --------------------------------------------------------------------------- #
# list_dir_files (private workspace helper)
# --------------------------------------------------------------------------- #


def test_list_dir_files_returns_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("code")
    (tmp_path / "notes.md").write_text("notes")
    files = list_dir_files(tmp_path)
    assert "a/x.py" in files
    assert "notes.md" in files


def test_list_dir_files_empty_dir(tmp_path: Path) -> None:
    assert list_dir_files(tmp_path) == []


def test_list_dir_files_missing_dir(tmp_path: Path) -> None:
    assert list_dir_files(tmp_path / "no_such") == []


def test_list_dir_files_respects_limit(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"file{i}.txt").write_text("x")
    result = list_dir_files(tmp_path, limit=3)
    assert len(result) == 3


# --------------------------------------------------------------------------- #
# F7: private workspace routing
# --------------------------------------------------------------------------- #


def test_apply_reply_routes_private_prefix(tmp_path: Path) -> None:
    """file:private/... blocks are written to private_root, not shared/."""
    ws = SharedWorkspace(tmp_path)
    private_root = tmp_path / "private"
    private_root.mkdir()

    text = "note\n```file:private/scratch.md\nhello\n```\n"
    writes = ws.apply_reply(text, private_root=private_root)

    assert len(writes) == 1
    assert writes[0].path == "private/scratch.md"
    # Written to private_root, NOT to shared/
    assert (private_root / "scratch.md").read_text() == "hello\n"
    assert not (ws.shared / "private").exists()


def test_apply_reply_private_and_shared_mixed(tmp_path: Path) -> None:
    """Blocks with and without private/ prefix are routed correctly."""
    ws = SharedWorkspace(tmp_path)
    private_root = tmp_path / "private"
    private_root.mkdir()

    text = (
        "```file:report.md\npublic\n```\n"
        "```file:private/draft.md\nprivate\n```\n"
    )
    writes = ws.apply_reply(text, private_root=private_root)

    paths = {w.path for w in writes}
    assert "report.md" in paths
    assert "private/draft.md" in paths
    assert (ws.shared / "report.md").exists()
    assert (private_root / "draft.md").exists()
    assert not (ws.shared / "private").exists()


def test_apply_reply_no_private_root_treats_as_shared(tmp_path: Path) -> None:
    """Without private_root, file:private/... writes to shared/ as before."""
    ws = SharedWorkspace(tmp_path)
    text = "```file:private/foo.txt\ndata\n```\n"
    writes = ws.apply_reply(text)
    assert len(writes) == 1
    assert (ws.shared / "private" / "foo.txt").exists()


def test_shared_workspace_shared_dir_alias(tmp_path: Path) -> None:
    """shared_dir is an alias for shared."""
    ws = SharedWorkspace(tmp_path)
    assert ws.shared_dir == ws.shared


# --------------------------------------------------------------------------- #
# CheckpointManager
# --------------------------------------------------------------------------- #

from team.workspace import CheckpointManager


def test_checkpoint_create_skips_empty_workspace(tmp_path: Path) -> None:
    """create() returns None when shared/ has no files yet."""
    mgr = CheckpointManager(tmp_path)
    assert mgr.create(0, "alice") is None
    assert not mgr.checkpoints_dir.exists()


def test_checkpoint_create_snapshots_files(tmp_path: Path) -> None:
    """create() copies shared/ into checkpoints/ and returns the path."""
    mgr = CheckpointManager(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "result.md").write_text("v1")

    dest = mgr.create(1, "alice")

    assert dest is not None
    assert dest.is_dir()
    assert (dest / "result.md").read_text() == "v1"


def test_checkpoint_list_checkpoints(tmp_path: Path) -> None:
    """list_checkpoints() returns CheckpointInfo objects in creation order."""
    mgr = CheckpointManager(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "a.txt").write_text("x")

    mgr.create(0, "alice")
    (shared / "b.txt").write_text("y")
    mgr.create(1, "bob")

    items = mgr.list_checkpoints()
    assert len(items) == 2
    assert items[0].member == "alice"
    assert items[0].turn == 0
    assert items[1].member == "bob"
    assert items[1].turn == 1
    assert items[1].file_count == 2  # a.txt + b.txt


def test_checkpoint_restore_replaces_shared(tmp_path: Path) -> None:
    """restore() replaces shared/ with the checkpoint's file tree."""
    mgr = CheckpointManager(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "original.txt").write_text("v1")

    mgr.create(0, "alice")

    # Simulate a member overwriting the file.
    (shared / "original.txt").write_text("v2")
    (shared / "new_file.txt").write_text("extra")

    cp = mgr.list_checkpoints()[0]
    mgr.restore(cp.id)

    assert (shared / "original.txt").read_text() == "v1"
    assert not (shared / "new_file.txt").exists()


def test_checkpoint_restore_unknown_id_raises(tmp_path: Path) -> None:
    """restore() raises ValueError for an unknown checkpoint ID."""
    mgr = CheckpointManager(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        mgr.restore("9999_nobody_20000101T000000")


def test_checkpoint_list_empty_when_no_dir(tmp_path: Path) -> None:
    """list_checkpoints() returns [] when no checkpoints directory exists."""
    mgr = CheckpointManager(tmp_path)
    assert mgr.list_checkpoints() == []

