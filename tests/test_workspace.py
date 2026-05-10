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

