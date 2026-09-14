"""``spec-kitty migrate backfill-merge-commit`` tests (issue #4231).

The backfill path for missions whose PR already merged before the
accept-side recording existed: the operator supplies the PR merge commit,
the migration verifies it against git (no fabrication — it must carry the
mission corpus, introduce it, and have LANDED on the target branch, so an
unmerged mission-branch commit is refused), and records
``baseline_merge_commit`` + ``pr_merge_commit`` through the same canonical
seam ``accept --merge-commit`` uses. Idempotent — an already-recorded
baseline is never overwritten.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import typer

from specify_cli.cli.commands.migrate_cmd import backfill_merge_commit_cmd

pytestmark = [pytest.mark.non_sandbox, pytest.mark.git_repo]

_SLUG = "320-pr-merged-mission-01TESTTE"
_MISSION_ID = "01TESTMIGRATEBACKFILL000000"


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _pr_merged_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    """A minimal repo: a mission merged into main by a PR-shaped merge.

    The mission needs only an identity-carrying ``meta.json`` — the backfill
    reads and writes that file; it does not re-run acceptance.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.name", "Backfill Test")
    _git(repo_root, "config", "user.email", "backfill-test@example.invalid")
    _git(repo_root, "branch", "-M", "main")
    (repo_root / ".kittify").mkdir()
    (repo_root / "README.md").write_text("# base\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "base")

    _git(repo_root, "checkout", "-qb", "kitty/mission-head")
    feature_dir = repo_root / "kitty-specs" / _SLUG
    feature_dir.mkdir(parents=True)
    (feature_dir / "meta.json").write_text(
        json.dumps(
            {
                "mission_id": _MISSION_ID,
                "mission_slug": _SLUG,
                "slug": _SLUG,
                "acceptance_mode": "pr",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "mission corpus")
    # A follow-up commit so the branch HEAD's parent already carries the
    # corpus (the mission-head confusion the verifier must reject).
    (feature_dir / "spec.md").write_text("# spec\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "mission follow-up")

    _git(repo_root, "checkout", "-q", "main")
    _git(
        repo_root,
        "merge",
        "--no-ff",
        "-qm",
        f"Merge PR: land {_SLUG}",
        "kitty/mission-head",
    )
    merge_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    pre_merge_parent = _git(repo_root, "rev-parse", "HEAD^1").stdout.strip()
    return repo_root, feature_dir, merge_commit, pre_merge_parent


def _run(
    repo_root: Path,
    merge_commit: str,
    *,
    dry_run: bool = False,
    target_branch: str | None = None,
) -> tuple[object, dict]:
    """Invoke the command directly; return (exit-or-None, parsed JSON payload)."""
    import contextlib
    import io

    stdout = io.StringIO()
    exit_obj: object = None
    with contextlib.redirect_stdout(stdout):
        try:
            backfill_merge_commit_cmd(
                mission=_SLUG,
                merge_commit=merge_commit,
                target_branch=target_branch,
                dry_run=dry_run,
                json_output=True,
            )
        except typer.Exit as exc:
            exit_obj = exc
    payload = json.loads(stdout.getvalue())
    return exit_obj, payload


def test_backfill_records_verified_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root, feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)

    exit_obj, payload = _run(repo_root, merge_commit)

    assert exit_obj is None, f"live run must exit 0, got {exit_obj!r}"
    row = payload["results"][0]
    assert row["action"] == "wrote"
    assert row["pr_merge_commit"] == merge_commit
    assert row["baseline_merge_commit"] == pre_merge_parent
    meta = json.loads((feature_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["baseline_merge_commit"] == pre_merge_parent
    assert meta["pr_merge_commit"] == merge_commit


def test_backfill_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root, feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)
    before = (feature_dir / "meta.json").read_text(encoding="utf-8")

    exit_obj, payload = _run(repo_root, merge_commit, dry_run=True)

    assert exit_obj is None
    row = payload["results"][0]
    assert row["action"] == "would_write"
    assert row["baseline_merge_commit"] == pre_merge_parent
    assert (feature_dir / "meta.json").read_text(encoding="utf-8") == before


def test_backfill_is_idempotent_skip_on_recorded_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root, feature_dir, merge_commit, _parent = _pr_merged_repo(tmp_path)
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)
    _run(repo_root, merge_commit)
    recorded = (feature_dir / "meta.json").read_text(encoding="utf-8")

    exit_obj, payload = _run(repo_root, merge_commit)

    assert exit_obj is None, "a skip is a success, not an error"
    row = payload["results"][0]
    assert row["action"] == "skip"
    assert "already recorded" in row["reason"]
    assert (feature_dir / "meta.json").read_text(encoding="utf-8") == recorded


def test_backfill_unverifiable_commit_fails_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root, feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)
    before = (feature_dir / "meta.json").read_text(encoding="utf-8")

    exit_obj, payload = _run(repo_root, "deadbeef" * 5)

    assert isinstance(exit_obj, typer.Exit)
    assert exit_obj.exit_code == 1  # type: ignore[attr-defined]
    row = payload["results"][0]
    assert row["action"] == "error"
    assert "cannot resolve" in row["reason"]
    assert (feature_dir / "meta.json").read_text(encoding="utf-8") == before


def test_backfill_rejects_mission_branch_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mission branch head is not merge evidence — the backfill refuses it."""
    repo_root, feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)
    mission_head = _git(repo_root, "rev-parse", "kitty/mission-head").stdout.strip()

    exit_obj, payload = _run(repo_root, mission_head)

    assert isinstance(exit_obj, typer.Exit)
    assert exit_obj.exit_code == 1  # type: ignore[attr-defined]
    row = payload["results"][0]
    assert row["action"] == "error"
    assert "not the landing commit" in row["reason"]


def _unmerged_specify_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    """The #4231 monitor repro at the CLI level: a Specify-only commit, unmerged.

    The mission branch carries exactly one commit introducing only
    ``kitty-specs/<slug>/meta.json`` and is NEVER merged into ``main``; the
    checkout stands on the mission branch (so the mission resolves and the
    skip-check can read its meta). Every first-introduction check passes for
    that commit — corpus at it, absent at its parent — so only the
    target-branch landing check can refuse it.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.name", "Backfill Test")
    _git(repo_root, "config", "user.email", "backfill-test@example.invalid")
    _git(repo_root, "branch", "-M", "main")
    (repo_root / ".kittify").mkdir()
    (repo_root / "README.md").write_text("# base\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "base")

    _git(repo_root, "checkout", "-qb", "kitty/mission-unmerged")
    feature_dir = repo_root / "kitty-specs" / _SLUG
    feature_dir.mkdir(parents=True)
    (feature_dir / "meta.json").write_text(
        json.dumps(
            {
                "mission_id": _MISSION_ID,
                "mission_slug": _SLUG,
                "slug": _SLUG,
                "acceptance_mode": "pr",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "Specify: mission corpus only")
    specify_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    return repo_root, feature_dir, specify_commit


def test_backfill_rejects_unmerged_specify_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unmerged Specify-only commit is refused through the public path.

    First introduction is not landing: the commit never reached the target
    branch, so recording it would anchor the dead-code scan at a tip the
    mission never reached. The refusal is clean (exit 1, nothing written).
    """
    repo_root, feature_dir, specify_commit = _unmerged_specify_repo(tmp_path)
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)
    before = (feature_dir / "meta.json").read_text(encoding="utf-8")

    exit_obj, payload = _run(repo_root, specify_commit)

    assert isinstance(exit_obj, typer.Exit)
    assert exit_obj.exit_code == 1  # type: ignore[attr-defined]
    row = payload["results"][0]
    assert row["action"] == "error"
    assert "not on target branch" in row["reason"]
    assert (feature_dir / "meta.json").read_text(encoding="utf-8") == before


def test_backfill_target_branch_override_for_non_primary_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PR that targeted a non-primary branch needs --target-branch.

    The mission declares no ``target_branch`` and the landing is on
    ``release`` while the repository's primary branch is ``main``: the
    auto-resolved target refuses the genuine landing (fail-closed, never a
    guessed branch), and the explicit ``--target-branch release`` records it.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.name", "Backfill Test")
    _git(repo_root, "config", "user.email", "backfill-test@example.invalid")
    _git(repo_root, "branch", "-M", "main")
    (repo_root / ".kittify").mkdir()
    (repo_root / "README.md").write_text("# base\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "base")
    _git(repo_root, "checkout", "-qb", "release")

    feature_dir = repo_root / "kitty-specs" / _SLUG
    feature_dir.mkdir(parents=True)
    (feature_dir / "meta.json").write_text(
        json.dumps(
            {
                "mission_id": _MISSION_ID,
                "mission_slug": _SLUG,
                "slug": _SLUG,
                "acceptance_mode": "pr",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "land on release")
    landing = _git(repo_root, "rev-parse", "release").stdout.strip()
    pre_release_tip = _git(repo_root, "rev-parse", "release^1").stdout.strip()
    monkeypatch.setenv("SPECIFY_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)

    # Auto-resolved target is the primary branch (main): the genuine landing
    # on release is refused — never silently re-targeted.
    exit_obj, payload = _run(repo_root, landing)
    assert isinstance(exit_obj, typer.Exit)
    row = payload["results"][0]
    assert row["action"] == "error"
    assert "not on target branch 'main'" in row["reason"]

    # The explicit PR base branch records it.
    exit_obj, payload = _run(repo_root, landing, target_branch="release")
    assert exit_obj is None, f"override run must exit 0, got {exit_obj!r}"
    row = payload["results"][0]
    assert row["action"] == "wrote"
    assert row["baseline_merge_commit"] == pre_release_tip
    meta = json.loads((feature_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["baseline_merge_commit"] == pre_release_tip
    assert meta["pr_merge_commit"] == landing
