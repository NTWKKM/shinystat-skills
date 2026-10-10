"""
tests/unit/test_shinystat_skill.py: Multi-agent mirror parity and schema test for shinystat skill.
"""

from pathlib import Path

import yaml


def test_shinystat_canonical_skill_exists_and_valid():
    """Verify skills/shinystat/SKILL.md exists, has valid YAML frontmatter and key sections."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    canonical_skill = repo_root / "skills" / "shinystat" / "SKILL.md"
    assert canonical_skill.exists(), "skills/shinystat/SKILL.md not found"

    content = canonical_skill.read_text(encoding="utf-8")
    assert content.startswith("---"), (
        "SKILL.md must start with YAML frontmatter delimiter"
    )

    parts = content.split("---", 2)
    assert len(parts) >= 3, "Frontmatter must be delimited by ---"
    frontmatter = yaml.safe_load(parts[1])

    assert frontmatter.get("name") == "shinystat"
    assert "description" in frontmatter
    assert len(frontmatter["description"]) < 1024

    # Body checks
    body = parts[2]
    assert "The 5 Decision Pillars" in body
    assert "The Deterministic Grilling Gate" in body
    assert "Completion Criteria" in body
    assert "references/decision-heuristics.md" in body
    assert "references/figures.md" in body
    assert "references/report-builder.md" in body
    assert "scripts/medstat" in body

    # Bundled scripts checks
    scripts_dir = canonical_skill.parent / "scripts"
    assert (scripts_dir / "medstat_cli.py").exists(), "scripts/medstat_cli.py missing"
    assert (scripts_dir / "medstat" / "cli" / "main.py").exists(), (
        "scripts/medstat/cli/main.py missing"
    )
    assert (scripts_dir / "medstat" / "models" / "firth.py").exists(), (
        "scripts/medstat/models/firth.py missing"
    )


def _is_valid_skill_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.name != ".DS_Store"
        and not path.name.endswith(".pyc")
        and "__pycache__" not in path.parts
    )


def test_shinystat_cloud_canonical_skill_exists_and_valid():
    """Verify skills/shinystat-cloud/SKILL.md exists, has valid YAML frontmatter, references, and matches packaging/cloud/shinystat-cloud."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    canonical_cloud = repo_root / "skills" / "shinystat-cloud"
    packaging_cloud = repo_root / "packaging" / "cloud" / "shinystat-cloud"

    cloud_skill = canonical_cloud / "SKILL.md"
    assert cloud_skill.exists(), "skills/shinystat-cloud/SKILL.md not found"

    content = cloud_skill.read_text(encoding="utf-8")
    assert content.startswith("---"), (
        "SKILL.md must start with YAML frontmatter delimiter"
    )

    parts = content.split("---", 2)
    assert len(parts) >= 3, "Frontmatter must be delimited by ---"
    frontmatter = yaml.safe_load(parts[1])

    assert frontmatter.get("name") == "shinystat-cloud"
    assert "description" in frontmatter
    assert len(frontmatter["description"]) < 1024

    # Body checks
    body = parts[2]
    assert "The 5 Decision Pillars" in body
    assert "references/decision-heuristics.md" in body
    assert "references/figures.md" in body
    assert "references/python-recipes.md" in body
    assert "references/report-builder.md" in body

    # References existence check
    ref_dir = cloud_skill.parent / "references"
    assert (ref_dir / "decision-heuristics.md").exists()
    assert (ref_dir / "figures.md").exists()
    assert (ref_dir / "python-recipes.md").exists()
    assert (ref_dir / "report-builder.md").exists()

    # Full tree and content parity with packaging/cloud/shinystat-cloud
    assert packaging_cloud.exists(), "packaging/cloud/shinystat-cloud not found"
    canonical_files = {
        path.relative_to(canonical_cloud)
        for path in canonical_cloud.rglob("*")
        if _is_valid_skill_file(path)
    }
    packaging_files = {
        path.relative_to(packaging_cloud)
        for path in packaging_cloud.rglob("*")
        if _is_valid_skill_file(path)
    }
    assert canonical_files == packaging_files, (
        f"Tree mismatch between skills/shinystat-cloud and packaging/cloud/shinystat-cloud: "
        f"missing={sorted(canonical_files - packaging_files)}, "
        f"extra={sorted(packaging_files - canonical_files)}"
    )

    for rel_path in canonical_files:
        canon_file = canonical_cloud / rel_path
        pack_file = packaging_cloud / rel_path
        assert pack_file.read_bytes() == canon_file.read_bytes(), (
            f"Byte mismatch in {rel_path} between skills/shinystat-cloud and packaging/cloud/shinystat-cloud"
        )


def test_shinystat_reference_manual_exists():
    """Verify skills/shinystat/references/decision-heuristics.md exists and contains archetype mappings."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    ref_doc = (
        repo_root / "skills" / "shinystat" / "references" / "decision-heuristics.md"
    )
    assert ref_doc.exists(), "references/decision-heuristics.md not found"
    ref_content = ref_doc.read_text(encoding="utf-8")
    assert "Clinical Study Archetype to Statistical Model Mapping" in ref_content
    assert "Events Per Variable (EPV)" in ref_content


def test_shinystat_multi_agent_mirrors_byte_identical():
    """Verify exact byte-identical parity between canonical skills/ root and all agent mirrors."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    canonical_dir = repo_root / "skills"

    mirrors = [
        repo_root / ".agent" / "skills",
        repo_root / ".agents" / "skills",
        repo_root / ".claude" / "skills",
        repo_root / ".cursor" / "skills",
    ]

    canonical_files = {
        path.relative_to(canonical_dir)
        for path in canonical_dir.rglob("*")
        if _is_valid_skill_file(path)
    }
    assert canonical_files, "No files found in canonical skills directory"

    for mirror in mirrors:
        assert mirror.exists(), f"Mirror {mirror} missing"
        mirror_files = {
            path.relative_to(mirror)
            for path in mirror.rglob("*")
            if _is_valid_skill_file(path)
        }
        assert mirror_files == canonical_files, (
            f"Path mismatch for mirror {mirror}: "
            f"missing={sorted(canonical_files - mirror_files)}, "
            f"extra={sorted(mirror_files - canonical_files)}"
        )
        for rel_path in canonical_files:
            canon_file = canonical_dir / rel_path
            mirror_file = mirror / rel_path
            assert mirror_file.read_bytes() == canon_file.read_bytes(), (
                f"Byte mismatch in {rel_path} for mirror {mirror}"
            )
