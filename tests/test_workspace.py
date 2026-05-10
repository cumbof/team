from pathlib import Path

import pytest

from team.workspace import SharedWorkspace, parse_file_blocks


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
