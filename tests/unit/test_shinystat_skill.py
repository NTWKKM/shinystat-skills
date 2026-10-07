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
    assert "The 4 Decision Pillars" in body
    assert "The Deterministic Grilling Gate" in body
    assert "Completion Criteria" in body
    assert "references/decision-heuristics.md" in body


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
        if path.is_file() and path.name != ".DS_Store"
    }
    assert canonical_files, "No files found in canonical skills directory"

    for mirror in mirrors:
        assert mirror.exists(), f"Mirror {mirror} missing"
        mirror_files = {
            path.relative_to(mirror)
            for path in mirror.rglob("*")
            if path.is_file() and path.name != ".DS_Store"
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
