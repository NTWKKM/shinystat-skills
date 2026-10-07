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
    """Verify exact byte-identical parity between canonical skills/shinystat and all agent mirrors."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    canonical_dir = repo_root / "skills" / "shinystat"

    mirrors = [
        repo_root / ".agent" / "skills" / "shinystat",
        repo_root / ".agents" / "skills" / "shinystat",
        repo_root / ".claude" / "skills" / "shinystat",
        repo_root / ".cursor" / "skills" / "shinystat",
    ]

    canonical_files = list(canonical_dir.rglob("*"))
    assert len(canonical_files) > 0

    for mirror in mirrors:
        assert mirror.exists(), f"Mirror {mirror} missing"
        for canon_file in canonical_files:
            if canon_file.is_file():
                rel_path = canon_file.relative_to(canonical_dir)
                mirror_file = mirror / rel_path
                assert mirror_file.exists(), (
                    f"File {rel_path} missing in mirror {mirror}"
                )
                assert mirror_file.read_bytes() == canon_file.read_bytes(), (
                    f"Byte mismatch in {rel_path} for mirror {mirror}"
                )
