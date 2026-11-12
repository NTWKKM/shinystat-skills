"""
tests/unit/test_install_skills.py: Unit tests for scripts/install-skills.sh
Validates legacy mirror cleanup, argument validation, dry-run mode, and scoped global installs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def fake_repo(tmp_path: Path) -> dict[str, Path]:
    """Scaffold a minimal isolated repository and HOME environment."""
    repo_dir = tmp_path / "fake_repo"
    home_dir = tmp_path / "fake_home"
    repo_dir.mkdir(parents=True)
    home_dir.mkdir(parents=True)

    # 1. Copy installer scripts
    scripts_dir = repo_dir / "scripts"
    scripts_dir.mkdir()
    shutil.copy(REPO_ROOT / "scripts" / "install-skills.sh", scripts_dir)
    shutil.copy(REPO_ROOT / "scripts" / "package-cloud-skill.sh", scripts_dir)
    shutil.copy(REPO_ROOT / "scripts" / "sync-skills.sh", scripts_dir)
    (scripts_dir / "install-skills.sh").chmod(0o755)
    (scripts_dir / "package-cloud-skill.sh").chmod(0o755)
    (scripts_dir / "sync-skills.sh").chmod(0o755)

    # 2. Canonical skills structure
    (repo_dir / "skills" / "shinystat" / "references").mkdir(parents=True)
    (repo_dir / "skills" / "shinystat" / "scripts").mkdir(parents=True)
    (repo_dir / "skills" / "shinystat" / "SKILL.md").write_text(
        "# Shinystat", encoding="utf-8"
    )

    (repo_dir / "skills" / "shinystat-cloud" / "references" / "recipes").mkdir(
        parents=True
    )
    (repo_dir / "skills" / "shinystat-cloud" / "SKILL.md").write_text(
        "# Shinystat Cloud", encoding="utf-8"
    )
    (repo_dir / "skills" / "shinystat-cloud" / "references" / "figures.md").write_text(
        "# Figures Cloud", encoding="utf-8"
    )

    (repo_dir / "skills" / "_shared" / "references").mkdir(parents=True)
    (
        repo_dir / "skills" / "_shared" / "references" / "decision-heuristics.md"
    ).write_text("# Decisions", encoding="utf-8")
    (repo_dir / "skills" / "_shared" / "references" / "figures.md").write_text(
        "# Figures Shared", encoding="utf-8"
    )

    # 3. Canonical src/medstat
    (repo_dir / "src" / "medstat").mkdir(parents=True)
    (repo_dir / "src" / "medstat" / "__init__.py").write_text(
        "# medstat", encoding="utf-8"
    )

    # 4. Canonical agent
    (repo_dir / "agents").mkdir(parents=True)
    (repo_dir / "agents" / "figure-tester.md").write_text(
        "# Figure Tester Agent", encoding="utf-8"
    )

    return {"repo": repo_dir, "home": home_dir}


def run_installer(
    repo: Path, home: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    """Execute install-skills.sh within the isolated fake repo and home."""
    env = {
        **os.environ,
        "HOME": str(home),
    }
    cmd = ["bash", str(repo / "scripts" / "install-skills.sh"), *args]
    return subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_user_skill_preserved_and_legacy_mirror_cleaned(fake_repo: dict[str, Path]):
    """Ensure user skills (.claude/skills/my-skill) remain intact while legacy mirrors are removed."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    claude_skills = repo / ".claude" / "skills"
    user_skill_dir = claude_skills / "my-skill"
    user_skill_dir.mkdir(parents=True)
    (user_skill_dir / "SKILL.md").write_text("# My Custom Skill", encoding="utf-8")

    legacy_mirror = claude_skills / "shinystat"
    legacy_mirror.mkdir(parents=True)
    (legacy_mirror / "SKILL.md").write_text("# Legacy Mirror", encoding="utf-8")

    res = run_installer(repo, home)
    assert res.returncode == 0, f"Installer failed: {res.stderr}"

    # User skill must survive untouched
    assert (user_skill_dir / "SKILL.md").is_file()
    assert (user_skill_dir / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "# My Custom Skill"
    assert f"[skip] {user_skill_dir} (not managed by shinystat-skills)" in res.stdout

    # Legacy mirror must be deleted
    assert not legacy_mirror.exists()
    assert f"[remove] {legacy_mirror}" in res.stdout


def test_symlink_inside_repo_deleted_outside_preserved(
    fake_repo: dict[str, Path], tmp_path: Path
):
    """Symlinks pointing inside repo are removed; symlinks pointing outside are preserved."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    external_dir = tmp_path / "external_workspace" / "other-skill"
    external_dir.mkdir(parents=True)
    (external_dir / "SKILL.md").write_text("# External", encoding="utf-8")

    cursor_skills = repo / ".cursor" / "skills"
    cursor_skills.mkdir(parents=True)

    inside_symlink = cursor_skills / "my_inside_link"
    os.symlink(repo / "skills" / "shinystat", inside_symlink)

    outside_symlink = cursor_skills / "my_outside_link"
    os.symlink(external_dir, outside_symlink)

    res = run_installer(repo, home)
    assert res.returncode == 0, f"Installer failed: {res.stderr}"

    # Symlink pointing into repo must be removed
    assert not inside_symlink.exists()
    assert not os.path.islink(inside_symlink)
    assert f"[remove] {inside_symlink}" in res.stdout

    # Symlink pointing outside repo must be preserved
    assert os.path.islink(outside_symlink)
    assert (outside_symlink / "SKILL.md").is_file()
    assert f"[skip] {outside_symlink} (not managed by shinystat-skills)" in res.stdout


def test_dry_run_performs_zero_deletion(fake_repo: dict[str, Path]):
    """--dry-run outputs deletion candidates without touching the filesystem."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    claude_skills = repo / ".claude" / "skills"
    legacy_dir = claude_skills / "shinystat"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "SKILL.md").write_text("# To Delete", encoding="utf-8")

    agent_dir = repo / ".agent" / "agents"
    agent_dir.mkdir(parents=True)
    legacy_agent = agent_dir / "figure-tester.md"
    legacy_agent.write_text("# Legacy Agent", encoding="utf-8")

    res = run_installer(repo, home, "--dry-run")
    assert res.returncode == 0, f"Installer failed: {res.stderr}"

    # Dry-run output emitted
    assert f"[dry-run] remove {legacy_dir}" in res.stdout
    assert f"[dry-run] remove {legacy_agent}" in res.stdout

    # Files must still exist
    assert legacy_dir.is_dir()
    assert legacy_agent.is_file()


def test_global_target_claude_isolated(fake_repo: dict[str, Path]):
    """--global --target claude only affects $HOME/.claude/skills and preserves user skills."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    user_claude_skill = home / ".claude" / "skills" / "my-favorite-tool"
    user_claude_skill.mkdir(parents=True)
    (user_claude_skill / "SKILL.md").write_text("# Keep Me", encoding="utf-8")

    res = run_installer(repo, home, "--global", "--target", "claude")
    assert res.returncode == 0, f"Installer failed: {res.stderr}"

    # shinystat and shinystat-cloud installed
    assert (home / ".claude" / "skills" / "shinystat" / "SKILL.md").is_file()
    assert (home / ".claude" / "skills" / "shinystat-cloud" / "SKILL.md").is_file()

    # User skill preserved
    assert (user_claude_skill / "SKILL.md").is_file()
    assert (user_claude_skill / "SKILL.md").read_text(encoding="utf-8") == "# Keep Me"

    # Other harnesses not touched
    assert not (home / ".gemini").exists()
    assert not (home / ".cursor").exists()


def test_global_target_antigravity_and_cursor(fake_repo: dict[str, Path]):
    """--global targets select exactly the corresponding directories."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    # Target antigravity
    res_ag = run_installer(repo, home, "--global", "-t", "antigravity")
    assert res_ag.returncode == 0
    assert (home / ".gemini" / "config" / "skills" / "shinystat" / "SKILL.md").is_file()
    assert (
        home / ".gemini" / "antigravity" / "skills" / "shinystat-cloud" / "SKILL.md"
    ).is_file()
    assert not (home / ".cursor").exists()
    assert not (home / ".claude").exists()

    # Target cursor
    res_cur = run_installer(repo, home, "--global", "--target", "cursor")
    assert res_cur.returncode == 0
    assert (home / ".cursor" / "skills" / "shinystat" / "SKILL.md").is_file()
    assert (home / ".cursor" / "skills" / "shinystat-cloud" / "SKILL.md").is_file()


def test_argument_validation_and_errors(fake_repo: dict[str, Path]):
    """Unknown arguments or malformed options trigger non-zero exit codes with error logs."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    # Unknown argument
    res1 = run_installer(repo, home, "--unknown-flag")
    assert res1.returncode != 0
    assert "Unknown argument" in res1.stderr

    # Missing value for -s
    res2 = run_installer(repo, home, "-s")
    assert res2.returncode != 0
    assert "Missing or invalid argument for -s" in res2.stderr

    # Option without value followed by another flag
    res3 = run_installer(repo, home, "--scope", "--target")
    assert res3.returncode != 0
    assert "Missing or invalid argument for --scope" in res3.stderr

    # Invalid scope value
    res4 = run_installer(repo, home, "-s", "invalid_scope")
    assert res4.returncode != 0
    assert "Invalid scope 'invalid_scope'" in res4.stderr

    # Invalid target value
    res5 = run_installer(repo, home, "-t", "invalid_target")
    assert res5.returncode != 0
    assert "Invalid target 'invalid_target'" in res5.stderr


def capture_workspace_state(repo: Path) -> dict[str, Any]:
    """Capture recursive snapshot of repo files, directories, symlink destinations, and content."""
    state = {}
    for path in sorted(repo.rglob("*")):
        rel_path = str(path.relative_to(repo))
        if path.is_symlink():
            state[rel_path] = ("symlink", os.readlink(path))
        elif path.is_file():
            state[rel_path] = ("file", path.read_text(encoding="utf-8"))
        elif path.is_dir():
            state[rel_path] = ("dir",)
    return state


def test_installer_is_idempotent(fake_repo: dict[str, Path]):
    """Running installer consecutively leaves an identical filesystem state and succeeds cleanly."""
    repo = fake_repo["repo"]
    home = fake_repo["home"]

    # Seed an unmanaged user skill and a legacy mirror before the first run
    unmanaged = repo / ".claude" / "skills" / "my-unmanaged"
    unmanaged.mkdir(parents=True)
    (unmanaged / "SKILL.md").write_text("# Unmanaged Skill", encoding="utf-8")

    legacy = repo / ".claude" / "skills" / "shinystat"
    legacy.mkdir(parents=True)
    (legacy / "SKILL.md").write_text("# Old Mirror", encoding="utf-8")

    # First run
    res_first = run_installer(repo, home)
    assert res_first.returncode == 0, f"First run failed: {res_first.stderr}"

    # Unmanaged skill survived, legacy mirror removed
    assert (unmanaged / "SKILL.md").is_file()
    assert not legacy.exists()

    # Capture workspace state after the first run
    state_first = capture_workspace_state(repo)

    # Second run
    res_second = run_installer(repo, home)
    assert res_second.returncode == 0, f"Second run failed: {res_second.stderr}"

    # Capture workspace state after the second run and assert zero changes
    state_second = capture_workspace_state(repo)
    assert state_second == state_first


def test_sync_check_reports_orphan_mirror_references(fake_repo: dict[str, Path]):
    """--check names a mirror reference with no shared source (ADR 69 Round 3)."""
    repo = fake_repo["repo"]
    sync = repo / "scripts" / "sync-skills.sh"
    local_refs = repo / "skills" / "shinystat" / "references"
    cloud_refs = repo / "skills" / "shinystat-cloud" / "references"

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(sync), *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run().returncode == 0
    # Files each mirror owns in place, with no source under _shared
    (local_refs / "api-quickref.md").write_text("# API", encoding="utf-8")
    (cloud_refs / "recipes" / "01-recipe.md").write_text("# Recipe", encoding="utf-8")
    assert run("--check").returncode == 0

    # Left behind after their canonical source was deleted
    (local_refs / "retired.md").write_text("# Retired", encoding="utf-8")
    (cloud_refs / "retired.md").write_text("# Retired", encoding="utf-8")
    check = run("--check")
    assert check.returncode == 1
    assert check.stderr.count("retired.md") >= 2

    # Syncing copies and never deletes: the orphans stay until someone removes them.
    assert run().returncode == 0
    assert (local_refs / "retired.md").exists()
    assert run("--check").returncode == 1

    (local_refs / "retired.md").unlink()
    (cloud_refs / "retired.md").unlink()
    assert run("--check").returncode == 0
    assert (local_refs / "api-quickref.md").exists()
    assert (cloud_refs / "figures.md").read_text(encoding="utf-8") == "# Figures Cloud"
    assert (cloud_refs / "recipes" / "01-recipe.md").exists()
