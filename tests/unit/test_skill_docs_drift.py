"""
Unit tests for Skill Documentation Parity, Schema Validity, and CLI Integrity.

Enforces ADR 8 and ADR 10:
1. Multi-agent mirror directories (.agent, .agents, .claude, .cursor) must be byte-identical to skills/.
2. YAML templates documented in SKILL.md/references must be valid under AnalysisPlan.from_yaml.
3. No phantom CLI flags or subcommands exist in documented bash snippets.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
import yaml

from medstat.cli.spec import AnalysisPlan

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_CANONICAL = REPO_ROOT / "skills"
MIRROR_DIRS = [
    REPO_ROOT / ".agent" / "skills",
    REPO_ROOT / ".agents" / "skills",
    REPO_ROOT / ".claude" / "skills",
    REPO_ROOT / ".cursor" / "skills",
]


def test_skills_canonical_exists():
    assert SKILLS_CANONICAL.is_dir()
    skills = [p.name for p in SKILLS_CANONICAL.iterdir() if p.is_dir()]
    assert "medstat-master" in skills
    assert len(skills) == 6


@pytest.mark.parametrize("mirror_dir", MIRROR_DIRS, ids=lambda d: d.parent.name)
def test_skill_mirror_parity(mirror_dir: Path):
    """Verify that all files in canonical skills/ are byte-identical in every mirror directory."""
    assert mirror_dir.is_dir(), f"Mirror directory missing: {mirror_dir}"

    canonical_files = sorted(
        [
            p.relative_to(SKILLS_CANONICAL)
            for p in SKILLS_CANONICAL.rglob("*")
            if p.is_file()
        ]
    )

    for rel_path in canonical_files:
        src = SKILLS_CANONICAL / rel_path
        dst = mirror_dir / rel_path
        assert dst.is_file(), (
            f"Missing file in mirror {mirror_dir.parent.name}: {rel_path}"
        )
        assert src.read_bytes() == dst.read_bytes(), (
            f"File mismatch between skills/ and {mirror_dir.parent.name} for {rel_path}"
        )


def test_autonomous_sap_yaml_template_parses():
    """Verify that the YAML template in autonomous-sap-template.md parses with AnalysisPlan."""
    template_md = (
        SKILLS_CANONICAL
        / "medstat-master"
        / "references"
        / "autonomous-sap-template.md"
    )
    assert template_md.is_file()

    content = template_md.read_text(encoding="utf-8")
    yaml_blocks = re.findall(r"```yaml\n(version:.*?)\n```", content, re.DOTALL)
    assert len(yaml_blocks) >= 1, "No version: 1.0 YAML block found in template"

    yaml_text = yaml_blocks[0]
    parsed_dict = yaml.safe_load(yaml_text)
    assert parsed_dict["version"] == "1.0"
    assert "metadata" in parsed_dict
    assert "data" in parsed_dict
    assert "variables" in parsed_dict
    assert "models" in parsed_dict

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_text)
        tmp_path = Path(tmp.name)

    try:
        plan = AnalysisPlan.from_yaml(tmp_path)
        assert plan.version == "1.0"
        assert plan.metadata.study_title == "Automated Clinical Cohort Analysis"
        assert len(plan.variables) >= 1
        assert len(plan.models) >= 1
    finally:
        tmp_path.unlink(missing_ok=True)


def test_no_phantom_cli_syntax_in_skills():
    """Verify that obsolete/phantom CLI commands and flags are absent from all skill markdown docs."""
    forbidden_patterns = [
        (r"\bmodel fit\b", "Phantom subcommand 'model fit' (use 'medstat model')"),
        (r"--y\s", "Phantom option '--y' (use '--outcome')"),
        (r"--x\s", "Phantom option '--x' (use '--covariates' or '--exposure')"),
        (r"--event\s", "Phantom option '--event' (use '--outcome')"),
        (
            r"\bmeta dl\b",
            "Phantom subcommand 'meta dl' (use 'medstat meta --method dl')",
        ),
        (
            r"medstat --spec\b",
            "Phantom root flag 'medstat --spec' (use 'medstat model --spec')",
        ),
    ]

    md_files = list(SKILLS_CANONICAL.rglob("*.md"))

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for pattern, msg in forbidden_patterns:
            matches = re.findall(pattern, text)
            assert not matches, (
                f"Forbidden pattern '{pattern}' ({msg}) found in {md_file.relative_to(REPO_ROOT)}"
            )
