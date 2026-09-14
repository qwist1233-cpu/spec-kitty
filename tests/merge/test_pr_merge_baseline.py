"""PR-merge evidence verification and recording (issue #4231).

A mission accepted through ``acceptance_mode: pr`` never passes through
``spec-kitty merge``, so its ``meta.json`` never carried
``baseline_merge_commit``. These tests pin the seam that repairs that state
from REAL git evidence only:

* :func:`verify_pr_merge_evidence` resolves the supplied commit, requires the
  mission corpus AT it and absent at its first parent (proving it is the
  landing commit — and rejecting the mission-branch-head confusion), and
  derives the pre-landing target tip from that parent.
* :func:`_record_pr_merge_baseline` writes ``baseline_merge_commit`` through
  the canonical writer plus the ``pr_merge_commit`` provenance field, and is
  idempotent over an already-recorded baseline.

Every test here drives the real ``git`` binary, so the module carries
``git_repo`` / ``integration`` / ``non_sandbox`` and must NOT carry ``fast``
(see ``tests/architectural/test_pytest_marker_correctness.py``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from specify_cli.merge.baseline import (
    _PR_MERGE_COMMIT_FIELD,
    _record_pr_merge_baseline,
    PrMergeEvidenceError,
    verify_pr_merge_evidence,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.git_repo,
    pytest.mark.non_sandbox,
]

_SLUG = "321-pr-merged-mission-01TESTTES"
_MISSION_ID = "01TESTPRMERGEEVIDENCE000000"


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_meta(feature_dir: Path) -> None:
    feature_dir.mkdir(parents=True)
    (feature_dir / "meta.json").write_text(
        json.dumps({"mission_id": _MISSION_ID, "mission_slug": _SLUG}) + "\n",
        encoding="utf-8",
    )


def _pr_merged_repo(
    tmp_path: Path,
    *,
    squash: bool = False,
) -> tuple[Path, Path, str, str]:
    """Build a repo whose mission landed on main through a PR-shaped merge.

    Returns ``(repo_root, feature_dir, merge_commit, pre_merge_parent)``. The
    base commit on main predates the mission corpus; the mission branch adds
    ``kitty-specs/<slug>/``; the landing commit merges it back — a two-parent
    merge commit by default, or a single-parent squash-style commit.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.name", "PR Baseline Test")
    _git(repo_root, "config", "user.email", "pr-baseline-test@example.invalid")
    _git(repo_root, "branch", "-M", "main")
    (repo_root / "README.md").write_text("# base\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-qm", "base")

    _git(repo_root, "checkout", "-qb", "kitty/mission-head")
    feature_dir = repo_root / "kitty-specs" / _SLUG
    _write_meta(feature_dir)
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "mission corpus")
    # A second commit on the mission branch, so the branch HEAD is a
    # follow-up whose parent already carries the corpus (the shape a real
    # mission branch has by the time its PR merges).
    (feature_dir / "spec.md").write_text("# spec\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "mission follow-up")

    _git(repo_root, "checkout", "-q", "main")
    if squash:
        # A squash-style landing: a single-parent commit on main that
        # introduces the mission corpus whose parent lacks it.
        _git(repo_root, "merge", "--squash", "-q", "kitty/mission-head")
        _git(repo_root, "commit", "-qm", "squash-land mission")
    else:
        _git(repo_root, "merge", "--no-ff", "-qm", "Merge PR: land mission", "kitty/mission-head")
    merge_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    pre_merge_parent = _git(repo_root, "rev-parse", "HEAD^1").stdout.strip()
    return repo_root, feature_dir, merge_commit, pre_merge_parent


def test_verify_two_parent_merge_commit_yields_pre_landing_tip(
    tmp_path: Path,
) -> None:
    """Happy path: baseline is the merge commit's first parent, not the merge itself."""
    repo_root, _feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)

    evidence = verify_pr_merge_evidence(repo_root, _SLUG, merge_commit)

    assert evidence.pr_merge_commit == merge_commit
    assert evidence.baseline_merge_commit == pre_merge_parent


def test_verify_accepts_short_sha_and_normalizes_to_full(
    tmp_path: Path,
) -> None:
    """A shortened SHA resolves to the full commit — the operator copies it off the PR page."""
    repo_root, _feature_dir, merge_commit, _parent = _pr_merged_repo(tmp_path)

    evidence = verify_pr_merge_evidence(repo_root, _SLUG, merge_commit[:10])

    assert evidence.pr_merge_commit == merge_commit


def test_verify_squash_style_landing_commit(
    tmp_path: Path,
) -> None:
    """A single-parent squash landing verifies the same way: parent lacks the corpus."""
    repo_root, _feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path, squash=True)

    evidence = verify_pr_merge_evidence(repo_root, _SLUG, merge_commit)

    assert evidence.baseline_merge_commit == pre_merge_parent


def test_verify_rejects_unknown_commit(tmp_path: Path) -> None:
    repo_root, _feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)

    with pytest.raises(PrMergeEvidenceError, match="cannot resolve"):
        verify_pr_merge_evidence(repo_root, _SLUG, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


def test_verify_rejects_empty_commit(tmp_path: Path) -> None:
    repo_root, _feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)

    with pytest.raises(PrMergeEvidenceError, match="no merge commit"):
        verify_pr_merge_evidence(repo_root, _SLUG, "   ")


def test_verify_rejects_commit_without_the_mission_corpus(tmp_path: Path) -> None:
    """A real commit that never carried this mission is not merge evidence for it."""
    repo_root, _feature_dir, _merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)
    # The pre-merge parent is a real commit — but it predates the mission.
    with pytest.raises(PrMergeEvidenceError, match="does not carry"):
        verify_pr_merge_evidence(repo_root, _SLUG, pre_merge_parent)


def test_verify_rejects_the_mission_branch_head(tmp_path: Path) -> None:
    """The mission branch HEAD carries the corpus — and so does ITS parent.

    Supplying the mission head instead of the PR merge commit would anchor the
    dead-code diff at the wrong tip (only the follow-up commit's delta) and
    silently skip the mission's own earlier additions, so the
    parent-carries-corpus check refuses it. (A single-commit mission branch's
    first commit is graph-identical to a squash landing — a new commit whose
    parent lacks the corpus — and anchoring at its parent still covers the
    whole mission, so that shape is legitimately accepted.)
    """
    repo_root, _feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)
    mission_head = _git(repo_root, "rev-parse", "kitty/mission-head").stdout.strip()

    with pytest.raises(PrMergeEvidenceError, match="not the landing commit"):
        verify_pr_merge_evidence(repo_root, _SLUG, mission_head)


def test_record_writes_baseline_and_provenance(tmp_path: Path) -> None:
    repo_root, feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)

    evidence = _record_pr_merge_baseline(feature_dir, repo_root, _SLUG, merge_commit)

    assert evidence.baseline_merge_commit == pre_merge_parent
    meta = json.loads((feature_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["baseline_merge_commit"] == pre_merge_parent
    assert meta[_PR_MERGE_COMMIT_FIELD] == merge_commit


def test_record_is_idempotent_and_set_once(tmp_path: Path) -> None:
    """A re-run never overwrites a recorded baseline or provenance stamp."""
    repo_root, feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)
    _record_pr_merge_baseline(feature_dir, repo_root, _SLUG, merge_commit)

    # Re-running with the same verified commit changes nothing.
    _record_pr_merge_baseline(feature_dir, repo_root, _SLUG, merge_commit)
    meta = json.loads((feature_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["baseline_merge_commit"] == pre_merge_parent
    assert meta[_PR_MERGE_COMMIT_FIELD] == merge_commit

    # An already-recorded baseline is never replaced, even by a fresh call
    # whose verification would derive a different value (set-once semantics of
    # the canonical writer, which this seam must not bypass).
    meta["baseline_merge_commit"] = "cafe000000000000000000000000000000000000"
    meta[_PR_MERGE_COMMIT_FIELD] = "f00d0000000000000000000000000000000000000"
    (feature_dir / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    _record_pr_merge_baseline(feature_dir, repo_root, _SLUG, merge_commit)
    meta = json.loads((feature_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["baseline_merge_commit"] == "cafe000000000000000000000000000000000000"
    assert meta[_PR_MERGE_COMMIT_FIELD] == "f00d0000000000000000000000000000000000000"


def test_record_refuses_unverifiable_commit_without_writing(tmp_path: Path) -> None:
    """No fabrication: an unverifiable SHA leaves meta.json untouched."""
    repo_root, feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)
    before = (feature_dir / "meta.json").read_text(encoding="utf-8")

    with pytest.raises(PrMergeEvidenceError):
        _record_pr_merge_baseline(feature_dir, repo_root, _SLUG, "0123456789abcdef" * 5)

    assert (feature_dir / "meta.json").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("bogus", ["--help", "HEAD", "main", "kitty/mission-head", "z9"])
def test_verify_rejects_non_sha_input_before_any_git_run(tmp_path: Path, bogus: str) -> None:
    """Shape guard: a non-hex value never reaches a git argument list.

    A value beginning with ``-`` would be parsed as a git option; a ref name
    is not the commit SHA an operator copied off a merged PR. Both are
    refused before any subprocess runs (the repo root need not even exist).
    """
    repo_root = tmp_path / "not-even-a-repo"
    repo_root.mkdir()

    with pytest.raises(PrMergeEvidenceError, match="is not a commit SHA"):
        verify_pr_merge_evidence(repo_root, _SLUG, bogus)
