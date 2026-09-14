"""PR-merge evidence verification and recording (issue #4231).

A mission accepted through ``acceptance_mode: pr`` never passes through
``spec-kitty merge``, so its ``meta.json`` never carried
``baseline_merge_commit``. These tests pin the seam that repairs that state
from REAL git evidence only:

* :func:`verify_pr_merge_evidence` resolves the supplied commit, requires the
  mission corpus AT it and absent at its first parent (proving it is the
  landing commit — and rejecting the mission-branch-head confusion), derives
  the pre-landing target tip from that parent, and requires the commit to
  have LANDED on the target branch — presence at the commit with absence at
  the parent is first-introduction evidence, which an unmerged mission-branch
  commit satisfies just as well as a real landing, so the
  ``merge-base --is-ancestor`` membership check is what tells them apart.
* :func:`_record_pr_merge_baseline` writes ``baseline_merge_commit`` through
  the canonical writer plus the ``pr_merge_commit`` provenance field, and is
  idempotent over an already-recorded baseline.
* The landing shapes GitHub actually produces are all covered: a two-parent
  merge commit, a single-parent squash landing, a rebase-style replay, and a
  staged corpus-then-implementation stack.

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
    parent lacks the corpus — but it has NOT landed on the target branch, so
    the landing check refuses it; see
    :func:`test_verify_rejects_unmerged_specify_only_branch_commit`.)
    """
    repo_root, _feature_dir, _merge_commit, _parent = _pr_merged_repo(tmp_path)
    mission_head = _git(repo_root, "rev-parse", "kitty/mission-head").stdout.strip()

    with pytest.raises(PrMergeEvidenceError, match="not the landing commit"):
        verify_pr_merge_evidence(repo_root, _SLUG, mission_head)


def _unmerged_specify_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    """The #4231 monitor repro: a Specify-only commit on an UNMERGED branch.

    ``main`` carries a base commit; the mission branch adds exactly one commit
    introducing only ``kitty-specs/<slug>/meta.json``; the branch is never
    merged. Presence of the corpus at that commit and absence at its parent
    prove first introduction — but not landing, which is exactly the
    distinction the verifier must make.
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

    _git(repo_root, "checkout", "-qb", "kitty/mission-unmerged")
    feature_dir = repo_root / "kitty-specs" / _SLUG
    _write_meta(feature_dir)
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "Specify: mission corpus only")
    specify_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    _git(repo_root, "checkout", "-q", "main")
    return repo_root, feature_dir, specify_commit


def test_verify_rejects_unmerged_specify_only_branch_commit(tmp_path: Path) -> None:
    """First introduction is not landing: an unmerged Specify commit is refused.

    The commit carries the corpus, its parent lacks it — every
    first-introduction check passes — but it is not an ancestor of the target
    branch, so it is not merge evidence. Recording it would anchor the
    dead-code scan at a tip the mission never reached.
    """
    repo_root, _feature_dir, specify_commit = _unmerged_specify_repo(tmp_path)

    with pytest.raises(PrMergeEvidenceError, match="not on target branch"):
        verify_pr_merge_evidence(repo_root, _SLUG, specify_commit)


def test_record_refuses_unmerged_specify_commit_without_writing(tmp_path: Path) -> None:
    """The recording seam refuses the unmerged Specify commit and writes nothing.

    The meta.json lives on the unmerged branch (absent from the ``main``
    checkout), so a refusal must leave the branch's copy untouched — verified
    by reading it through git, not the (absent) working-tree path.
    """
    repo_root, feature_dir, specify_commit = _unmerged_specify_repo(tmp_path)
    before = _git(repo_root, "show", f"kitty/mission-unmerged:kitty-specs/{_SLUG}/meta.json").stdout

    with pytest.raises(PrMergeEvidenceError, match="not on target branch"):
        _record_pr_merge_baseline(feature_dir, repo_root, _SLUG, specify_commit)

    after = _git(repo_root, "show", f"kitty/mission-unmerged:kitty-specs/{_SLUG}/meta.json").stdout
    assert after == before


def _rebase_landed_repo(tmp_path: Path) -> tuple[Path, Path, str, str, str]:
    """A rebase-style PR landing: branch commits replayed onto the target tip.

    Returns ``(repo_root, feature_dir, replayed_corpus_commit,
    pre_landing_tip, original_corpus_commit)``. The branch carries
    [corpus][follow-up]; it lands by replaying onto ``main`` (the rebase-merge
    shape), so the replayed corpus commit's first parent IS the pre-landing
    ``main`` tip — while the ORIGINAL branch commit, graph-identical in
    content, never landed.
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
    original_corpus = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    (feature_dir / "spec.md").write_text("# spec\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "mission follow-up")

    # Advance main past the fork point so the replay is a real rebase (the
    # branch commits are re-created on the new tip), not a fast-forward.
    _git(repo_root, "checkout", "-q", "main")
    (repo_root / "unrelated.txt").write_text("other PR\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "unrelated work lands first")
    pre_landing_tip = _git(repo_root, "rev-parse", "main").stdout.strip()

    # Rebase-merge landing: replay the branch onto the new main tip, then
    # fast-forward main to the replayed commits.
    _git(repo_root, "checkout", "-q", "-b", "replay", "kitty/mission-head")
    _git(repo_root, "rebase", "main")
    _git(repo_root, "checkout", "-q", "main")
    _git(repo_root, "merge", "--ff-only", "-q", "replay")
    replayed_corpus = _git(repo_root, "rev-parse", "main~1").stdout.strip()
    _git(repo_root, "branch", "-qD", "replay")
    return repo_root, feature_dir, replayed_corpus, pre_landing_tip, original_corpus


def test_verify_accepts_rebase_style_landing(tmp_path: Path) -> None:
    """A rebase-merge landing verifies at the replayed corpus commit.

    Its first parent is the pre-landing target tip — the replay put the
    corpus commit directly onto the old ``main`` tip — and the follow-up
    commit on top of it is refused (its parent carries the corpus).
    """
    repo_root, _feature_dir, replayed_corpus, pre_landing_tip, _orig = _rebase_landed_repo(tmp_path)

    evidence = verify_pr_merge_evidence(repo_root, _SLUG, replayed_corpus)

    assert evidence.pr_merge_commit == replayed_corpus
    assert evidence.baseline_merge_commit == pre_landing_tip

    follow_up = _git(repo_root, "rev-parse", "main").stdout.strip()
    with pytest.raises(PrMergeEvidenceError, match="not the landing commit"):
        verify_pr_merge_evidence(repo_root, _SLUG, follow_up)


def test_verify_rejects_original_branch_commit_after_rebase_landing(tmp_path: Path) -> None:
    """The pre-replay branch commit is content-identical to the landing — and refused.

    Every first-introduction check passes for it (corpus at it, absent at its
    parent), but it never landed on the target branch: only the replayed
    copies did. This is the unmerged-commit distinction in its sharpest form.
    """
    repo_root, _feature_dir, _replayed, _tip, original_corpus = _rebase_landed_repo(tmp_path)

    with pytest.raises(PrMergeEvidenceError, match="not on target branch"):
        verify_pr_merge_evidence(repo_root, _SLUG, original_corpus)


def test_verify_accepts_staged_corpus_then_implementation(tmp_path: Path) -> None:
    """Corpus first, implementation after: the corpus commit is the landing.

    A staged workflow whose FIRST landed commit carries the whole mission
    corpus (``meta.json`` + spec) anchors at the pre-landing tip and covers
    the implementation commit on top of it.
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

    feature_dir = repo_root / "kitty-specs" / _SLUG
    _write_meta(feature_dir)
    (feature_dir / "spec.md").write_text("# spec\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "Specify: full corpus")
    corpus_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    pre_landing_tip = _git(repo_root, "rev-parse", "HEAD^1").stdout.strip()

    # The implementation commit lands on top of the corpus.
    (repo_root / "src_impl.py").write_text("IMPL = 1\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "implement")

    evidence = verify_pr_merge_evidence(repo_root, _SLUG, corpus_commit)

    assert evidence.baseline_merge_commit == pre_landing_tip


def test_verify_rejects_meta_landing_after_staged_siblings(tmp_path: Path) -> None:
    """A corpus staged spec-first, meta-last: the meta commit is refused.

    ``spec.md`` landed in an earlier commit, so the ``meta.json``-introducing
    commit's first parent already carries part of the mission corpus. A diff
    anchored there would skip the spec — so this is a refusal, never a
    guessed earlier anchor (the operator supplies the commit that introduced
    the corpus, or the mission re-lands whole).
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

    feature_dir = repo_root / "kitty-specs" / _SLUG
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "spec.md").write_text("# spec\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "Specify: spec only")
    spec_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    (feature_dir / "meta.json").write_text(
        json.dumps({"mission_id": _MISSION_ID, "mission_slug": _SLUG}) + "\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "meta lands later")
    meta_commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    # The spec-only commit does not carry meta.json at all.
    with pytest.raises(PrMergeEvidenceError, match="does not carry"):
        verify_pr_merge_evidence(repo_root, _SLUG, spec_commit)
    # The meta.json commit's parent already carries the mission directory.
    with pytest.raises(PrMergeEvidenceError, match="staged corpus"):
        verify_pr_merge_evidence(repo_root, _SLUG, meta_commit)


def test_verify_explicit_target_ref(tmp_path: Path) -> None:
    """An explicit target branch (any rev-parse form) is honored verbatim."""
    repo_root, _feature_dir, merge_commit, pre_merge_parent = _pr_merged_repo(tmp_path)

    for ref in ("main", "refs/heads/main"):
        evidence = verify_pr_merge_evidence(repo_root, _SLUG, merge_commit, target_ref=ref)
        assert evidence.baseline_merge_commit == pre_merge_parent


def test_verify_honors_mission_declared_target_branch(tmp_path: Path) -> None:
    """The mission's declared ``target_branch`` is the default landing target.

    A landing on a non-primary branch ``release`` verifies without any
    explicit ``--target-branch`` because the mission meta declares it — and
    the SAME commit is refused when the meta instead declares ``main`` (the
    declared target wins over the repository's primary branch).
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
    _git(repo_root, "checkout", "-qb", "release")

    feature_dir = repo_root / "kitty-specs" / _SLUG
    _write_meta(feature_dir)
    _meta = json.loads((feature_dir / "meta.json").read_text(encoding="utf-8"))
    _meta["target_branch"] = "release"
    (feature_dir / "meta.json").write_text(json.dumps(_meta) + "\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "land on release")
    landing = _git(repo_root, "rev-parse", "release").stdout.strip()
    pre_release_tip = _git(repo_root, "rev-parse", "release^1").stdout.strip()

    evidence = verify_pr_merge_evidence(repo_root, _SLUG, landing)
    assert evidence.baseline_merge_commit == pre_release_tip

    # Same commit, mission now declaring main: refused — the declared target
    # is the landing target, not whatever branch happens to hold the commit.
    (feature_dir / "meta.json").write_text(
        json.dumps({"mission_id": _MISSION_ID, "mission_slug": _SLUG, "target_branch": "main"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PrMergeEvidenceError, match="not on target branch 'main'"):
        verify_pr_merge_evidence(repo_root, _SLUG, landing)


def test_verify_rejects_unknown_target_branch(tmp_path: Path) -> None:
    """A target branch that does not exist locally is its own clean refusal."""
    repo_root, _feature_dir, merge_commit, _parent = _pr_merged_repo(tmp_path)

    with pytest.raises(PrMergeEvidenceError, match="cannot resolve"):
        verify_pr_merge_evidence(repo_root, _SLUG, merge_commit, target_ref="no-such-branch")


@pytest.mark.parametrize("bogus_target", ["", "   ", "--evil", "-b"])
def test_verify_rejects_malformed_target_ref(tmp_path: Path, bogus_target: str) -> None:
    """Shape guard: an empty or option-shaped target branch never reaches git."""
    repo_root, _feature_dir, merge_commit, _parent = _pr_merged_repo(tmp_path)

    with pytest.raises(PrMergeEvidenceError, match="not a branch name|is empty"):
        verify_pr_merge_evidence(repo_root, _SLUG, merge_commit, target_ref=bogus_target)


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
