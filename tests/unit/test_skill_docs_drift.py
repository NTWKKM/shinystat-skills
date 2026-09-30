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

    mirror_files = sorted(
        [p.relative_to(mirror_dir) for p in mirror_dir.rglob("*") if p.is_file()]
    )
    assert mirror_files == canonical_files, (
        f"Mirror directory {mirror_dir.parent.name} file set differs from canonical skills/:\n"
        f"Extra in mirror: {set(mirror_files) - set(canonical_files)}\n"
        f"Missing from mirror: {set(canonical_files) - set(mirror_files)}"
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
    import re
    import shlex

    import click

    from medstat.cli.main import cli

    md_files = list(SKILLS_CANONICAL.rglob("*.md"))

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")

        # Check both bash blocks and inline code blocks
        blocks = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        inline_blocks = re.findall(r"`([^`]+)`", text)

        lines_to_check = []
        for block in blocks:
            # Handle line continuations in bash blocks
            block = block.replace("\\\n", " ")
            lines_to_check.extend(block.split("\n"))
        lines_to_check.extend(inline_blocks)

        for line in lines_to_check:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if "medstat " in line:
                # Basic cleanup
                if line.startswith("uv run "):
                    line = line[len("uv run ") :]
                if line.startswith("$ "):
                    line = line[2:]

                try:
                    args = shlex.split(line)
                except ValueError:
                    continue

                if not args or args[0] != "medstat":
                    continue

                # Remove the first "medstat" since we start evaluating against the cli root group
                test_args = args[1:]

                current = cli
                idx = 0
                cmd_path = ["medstat"]

                while idx < len(test_args):
                    arg = test_args[idx]
                    if arg.startswith("-"):
                        break

                    if isinstance(current, click.Group):
                        ctx = click.Context(current)
                        cmd = current.get_command(ctx, arg)
                        assert cmd is not None, (
                            f"Phantom subcommand '{arg}' for '{' '.join(cmd_path)}' found in {md_file.relative_to(REPO_ROOT)}"
                        )
                        current = cmd
                        cmd_path.append(arg)
                        idx += 1
                    else:
                        # Leaf command, subsequent non-dash arguments are positional
                        break

                # Now validate options for `current`
                allowed_opts = set()
                for param in current.params:
                    if isinstance(param, click.Option):
                        allowed_opts.update(param.opts)
                        allowed_opts.update(param.secondary_opts)

                allowed_opts.update(["--help", "--version"])

                for arg in test_args[idx:]:
                    if arg.startswith("-"):
                        opt_name = arg.split("=")[0]
                        if opt_name.startswith("-") and not opt_name.startswith("--<"):
                            assert opt_name in allowed_opts, (
                                f"Phantom option '{opt_name}' for command '{' '.join(cmd_path)}' found in {md_file.relative_to(REPO_ROOT)}"
                            )
