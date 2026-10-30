"""
tests/unit/test_shinystat_skill.py: Multi-agent mirror parity and schema test for shinystat skill.
"""

import os
import subprocess
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

    # Runtime requirements stack check
    req_file = canonical_skill.parent / "requirements.txt"
    assert req_file.exists(), "skills/shinystat/requirements.txt missing"

    expected_packages = {
        "pandas",
        "openpyxl",
        "pyarrow",
        "numpy",
        "scipy",
        "statsmodels",
        "patsy",
        "scikit-learn",
        "lifelines",
        "firthmodels",
        "matplotlib",
        "plotly",
        "python-docx",
        "python-pptx",
        "click",
        "pyyaml",
    }

    import re

    parsed_deps = {}
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"Runtime dependency {line} must use exact '==' pin"
        pkg, sep, ver = line.partition("==")
        pkg = pkg.strip()
        ver = ver.strip()
        assert pkg, f"Package name cannot be empty in {line}"
        assert ver, f"Version pin cannot be empty for package {pkg}"
        assert re.match(r"^\d+(\.\d+)*(\.post\d+)?$", ver), (
            f"Invalid version string '{ver}' for package {pkg}"
        )
        parsed_deps[pkg] = ver

    assert set(parsed_deps.keys()) == expected_packages, (
        f"Mismatch in runtime packages: missing={expected_packages - set(parsed_deps.keys())}, "
        f"unexpected={set(parsed_deps.keys()) - expected_packages}"
    )

    req_text = req_file.read_text(encoding="utf-8")
    assert "--with playwright python -m playwright install chromium" in req_text
    assert "--with-deps" in req_text


def _is_valid_skill_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.name != ".DS_Store"
        and not path.name.endswith(".pyc")
        and "__pycache__" not in path.parts
    )


def _collect_valid_skill_files(root: Path) -> set[Path]:
    import os

    files = set()
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for f in filenames:
            p = Path(dirpath) / f
            if _is_valid_skill_file(p):
                files.add(p.relative_to(root))
    return files


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
    assert "_shared/references/decision-heuristics.md" in body
    assert "_shared/references/figures.md" in body
    assert "references/recipes/01-data-missing-table1.md" in body
    assert "_shared/references/report-builder.md" in body

    # References existence check
    shared_ref_dir = repo_root / "skills" / "_shared" / "references"
    assert (shared_ref_dir / "decision-heuristics.md").exists()
    assert (shared_ref_dir / "figures.md").exists()
    assert (shared_ref_dir / "report-builder.md").exists()

    recipes_dir = cloud_skill.parent / "references" / "recipes"
    for r_num in range(1, 6):
        pattern = f"0{r_num}-*.md"
        matched = list(recipes_dir.glob(pattern))
        assert len(matched) == 1, (
            f"Expected 1 recipe file matching {pattern}, found {len(matched)}"
        )

    # Ensure standalone cloud package is generated via the supported packaging workflow
    package_script = repo_root / "scripts" / "package-cloud-skill.sh"
    assert package_script.exists(), "scripts/package-cloud-skill.sh not found"
    subprocess.run(
        ["bash", str(package_script)], check=True, cwd=repo_root, capture_output=True
    )

    # Full tree and content parity with packaging/cloud/shinystat-cloud
    assert packaging_cloud.exists(), "packaging/cloud/shinystat-cloud not found"
    canonical_files = _collect_valid_skill_files(canonical_cloud)
    packaging_files = _collect_valid_skill_files(packaging_cloud)
    assert canonical_files == packaging_files, (
        f"Tree mismatch between skills/shinystat-cloud and packaging/cloud/shinystat-cloud: "
        f"missing={sorted(canonical_files - packaging_files)}, "
        f"extra={sorted(packaging_files - canonical_files)}"
    )

    for rel_path in canonical_files:
        canon_file = canonical_cloud / rel_path
        pack_file = packaging_cloud / rel_path
        if pack_file.read_bytes() == canon_file.read_bytes():
            continue
        # In packaged standalone distributions, relative links to _shared references
        # are rewritten to resolve to the packaged copies under references/
        if rel_path.suffix == ".md":
            ref_rel = os.path.relpath(packaging_cloud / "references", pack_file.parent)
            prefix = "" if ref_rel == "." else f"{ref_rel}/"
            expected = (
                canon_file.read_text(encoding="utf-8")
                .replace("../../../_shared/references/", prefix)
                .replace("../../_shared/references/", prefix)
                .replace("../_shared/references/", prefix)
                .replace("(`../_shared/references/`)", f"(`{prefix}`)")
            )
            assert pack_file.read_text(encoding="utf-8") == expected, (
                f"Content mismatch in {rel_path} between skills/shinystat-cloud and packaging/cloud/shinystat-cloud"
            )
        else:
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
        p for p in _collect_valid_skill_files(canonical_dir) if "_shared" not in p.parts
    }
    assert canonical_files, "No files found in canonical skills directory"

    for mirror in mirrors:
        assert mirror.exists(), f"Mirror {mirror} missing"
        mirror_files = _collect_valid_skill_files(mirror)
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


def test_src_medstat_and_skills_bundled_medstat_byte_identical():
    """Verify exact byte-identical parity between src/medstat (SSOT) and all bundled medstat targets."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src" / "medstat"
    assert src_dir.exists(), "src/medstat directory not found"

    targets = [
        (
            "canonical skills/",
            repo_root / "skills" / "shinystat" / "scripts" / "medstat",
        ),
        (
            "antigravity .agent mirror",
            repo_root / ".agent" / "skills" / "shinystat" / "scripts" / "medstat",
        ),
        (
            "antigravity .agents mirror",
            repo_root / ".agents" / "skills" / "shinystat" / "scripts" / "medstat",
        ),
        (
            "claude code .claude mirror",
            repo_root / ".claude" / "skills" / "shinystat" / "scripts" / "medstat",
        ),
        (
            "cursor .cursor mirror",
            repo_root / ".cursor" / "skills" / "shinystat" / "scripts" / "medstat",
        ),
    ]

    src_files = _collect_valid_skill_files(src_dir)
    assert src_files, "No valid files found in src/medstat"

    for label, target_dir in targets:
        assert target_dir.exists(), f"{label} target directory not found: {target_dir}"
        target_files = _collect_valid_skill_files(target_dir)
        assert src_files == target_files, (
            f"File tree mismatch between src/medstat and {label} ({target_dir}): "
            f"missing_in_target={sorted(src_files - target_files)}, "
            f"extra_in_target={sorted(target_files - src_files)}"
        )
        for rel_path in src_files:
            src_file = src_dir / rel_path
            target_file = target_dir / rel_path
            assert target_file.read_bytes() == src_file.read_bytes(), (
                f"Byte mismatch in {rel_path} between src/medstat and {label} ({target_dir})"
            )
