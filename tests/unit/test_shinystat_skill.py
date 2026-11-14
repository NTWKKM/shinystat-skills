"""
tests/unit/test_shinystat_skill.py: Multi-agent mirror parity and schema test for shinystat skill.
"""

import re
import subprocess
from pathlib import Path

import pytest
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
        "tabulate",  # render_md -> DataFrame.to_markdown
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


def _extract_heading_slugs(text: str) -> set[str]:
    """Generates heading anchor slugs following GitHub Markdown specifications."""
    # Exclude fenced code blocks before extracting headings
    clean_text = re.sub(
        r"(?ms)^(`{3,})[^\n]*\n(?:.*?\n)?\1`*[ \t]*$|^(~{3,})[^\n]*\n(?:.*?\n)?\2~*[ \t]*$",
        "",
        text,
    )
    doc_headings = re.findall(r"^#+\s+(.+)$", clean_text, flags=re.MULTILINE)
    slugs = set()
    seen: dict[str, int] = {}
    for h in doc_headings:
        h_clean = re.sub(r"\s*#+\s*$", "", h).strip()
        # Normalize Markdown links [text](url) -> text so URLs do not contribute to slug
        h_clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", h_clean)
        base = re.sub(
            r"\s", "-", re.sub(r"[^\w\s-]", "", re.sub(r"[`*]", "", h_clean).lower())
        )
        n = seen.get(base, 0)
        seen[base] = n + 1
        slugs.add(base if n == 0 else f"{base}-{n}")
    return slugs


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
    assert "references/recipes/01-data-missing-table1.md" in body
    assert "references/report-builder.md" in body

    cloud_figures = canonical_cloud / "references" / "figures.md"
    assert cloud_figures.exists(), (
        "skills/shinystat-cloud/references/figures.md missing"
    )
    assert "from medstat" not in cloud_figures.read_text(encoding="utf-8"), (
        "Cloud figures.md must not contain any medstat imports"
    )

    # References existence check: verify all references and fragments linked in body exist
    raw_ref_links = set(re.findall(r"\(references/([^)]+)\)", body))
    assert raw_ref_links, "Expected at least one reference linked in SKILL.md body"
    for link in sorted(raw_ref_links):
        rel_file, fragment = link.split("#", 1) if "#" in link else (link, None)
        target_path = canonical_cloud / "references" / rel_file
        assert target_path.exists(), (
            f"Linked reference '{rel_file}' missing from {canonical_cloud / 'references'}"
        )
        if fragment:
            slugs = _extract_heading_slugs(target_path.read_text(encoding="utf-8"))
            assert fragment in slugs, (
                f"Linked fragment '#{fragment}' not found in {rel_file} headings"
            )

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

    # Full tree and byte-identical parity with packaging/cloud/shinystat-cloud
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
        assert pack_file.read_bytes() == canon_file.read_bytes(), (
            f"Byte mismatch in {rel_path} between skills/shinystat-cloud and packaging/cloud/shinystat-cloud"
        )

    # Independent validation: verify every local relative Markdown link resolves to an
    # existing target strictly within packaging_cloud, without relying on string replacement
    md_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    resolved_links_count = 0
    for pack_file in packaging_cloud.rglob("*.md"):
        if not pack_file.is_file():
            continue
        text = pack_file.read_text(encoding="utf-8")
        for match in md_link_pattern.finditer(text):
            target = match.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_clean = target.split()[0]
            if target_clean.startswith("#"):
                fragment = target_clean[1:]
                target_file = pack_file
            else:
                rel_part, _, fragment = target_clean.partition("#")
                if not rel_part:
                    continue
                target_file = (pack_file.parent / rel_part).resolve()
                assert target_file.exists(), (
                    f"Broken relative link in {pack_file.relative_to(packaging_cloud)}: "
                    f"target '{target}' does not exist at {target_file}"
                )
                assert (
                    packaging_cloud.resolve() in target_file.parents
                    or target_file == packaging_cloud.resolve()
                ), (
                    f"Escaping relative link in {pack_file.relative_to(packaging_cloud)}: "
                    f"target '{target}' resolves outside packaging directory at {target_file}"
                )
            if fragment and target_file.suffix == ".md":
                slugs = _extract_heading_slugs(target_file.read_text(encoding="utf-8"))
                assert fragment in slugs, (
                    f"Linked fragment '#{fragment}' in {pack_file.relative_to(packaging_cloud)} "
                    f"not found in {target_file.relative_to(packaging_cloud)} headings"
                )
            resolved_links_count += 1

    assert resolved_links_count > 0, (
        "Expected at least one relative Markdown link in packaged cloud skill"
    )


def test_shinystat_local_skill_markdown_links_resolve():
    """Verify every relative Markdown link in skills/shinystat resolves to an existing file and valid heading slug."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_skill_dir = repo_root / "skills" / "shinystat"
    assert local_skill_dir.exists(), "skills/shinystat not found"

    md_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    resolved_links_count = 0
    for md_file in local_skill_dir.rglob("*.md"):
        if not md_file.is_file():
            continue
        text = md_file.read_text(encoding="utf-8")
        for match in md_link_pattern.finditer(text):
            target = match.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_clean = target.split()[0]
            if target_clean.startswith("#"):
                fragment = target_clean[1:]
                target_file = md_file
            else:
                rel_part, _, fragment = target_clean.partition("#")
                if not rel_part:
                    continue
                target_file = (md_file.parent / rel_part).resolve()
                assert target_file.exists(), (
                    f"Broken relative link in {md_file.relative_to(local_skill_dir)}: "
                    f"target '{target}' does not exist at {target_file}"
                )
                assert (
                    local_skill_dir.resolve() in target_file.parents
                    or target_file == local_skill_dir.resolve()
                ), (
                    f"Escaping relative link in {md_file.relative_to(local_skill_dir)}: "
                    f"target '{target}' resolves outside local skill directory at {target_file}"
                )
            if fragment and target_file.suffix == ".md":
                slugs = _extract_heading_slugs(target_file.read_text(encoding="utf-8"))
                assert fragment in slugs, (
                    f"Linked fragment '#{fragment}' in {md_file.relative_to(local_skill_dir)} "
                    f"not found in {target_file.relative_to(local_skill_dir)} headings"
                )
            resolved_links_count += 1

    assert resolved_links_count > 0, "Expected relative Markdown links in local skill"


def test_no_phantom_module_or_function_symbols_in_skill_docs():
    """Verify skill docs do not reference non-existent module or function names (W-1)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_skill = repo_root / "skills" / "shinystat" / "SKILL.md"
    decision_heuristics = (
        repo_root / "skills" / "_shared" / "references" / "decision-heuristics.md"
    )

    skill_text = local_skill.read_text(encoding="utf-8")
    heuristics_text = decision_heuristics.read_text(encoding="utf-8")

    # SKILL.md checks
    assert "medstat.diagnostics" not in skill_text, (
        "Found phantom 'medstat.diagnostics' (should be medstat.diagnostic)"
    )
    assert "(`survival`, `competing_risks`)" not in skill_text, (
        "Found phantom 'competing_risks' module in SKILL.md"
    )

    # decision-heuristics.md checks
    assert "fit_ordinal_logistic" not in heuristics_text, (
        "Found phantom 'fit_ordinal_logistic' (should be fit_proportional_odds)"
    )
    assert "fit_rcs_logistic" not in heuristics_text, (
        "Found phantom 'fit_rcs_logistic' (should be fit_logistic_rcs)"
    )
    assert "fit_rcs_cox" not in heuristics_text, (
        "Found phantom 'fit_rcs_cox' (should be fit_cox_rcs)"
    )
    assert "fit_clustered_gee" not in heuristics_text, (
        "Found phantom 'fit_clustered_gee' (should be fit_gee)"
    )
    assert "rcs_lib.py" not in heuristics_text, "Found phantom 'rcs_lib.py'"


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


def test_shinystat_canonical_shared_references_synced():
    """Verify canonical skills/shinystat and skills/shinystat-cloud contain synced shared references."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    shared_ref_dir = repo_root / "skills" / "_shared" / "references"
    local_ref_dir = repo_root / "skills" / "shinystat" / "references"
    cloud_ref_dir = repo_root / "skills" / "shinystat-cloud" / "references"

    shared_files = {p.name for p in shared_ref_dir.glob("*.md") if p.is_file()}
    assert shared_files, "No shared reference files found in skills/_shared/references"

    for fname in sorted(shared_files):
        shared_file = shared_ref_dir / fname
        local_file = local_ref_dir / fname
        assert local_file.exists(), (
            f"Missing shared reference in local skill: {local_file}"
        )
        assert local_file.read_bytes() == shared_file.read_bytes(), (
            f"Byte mismatch in {fname} between skills/_shared/references and skills/shinystat/references"
        )
        if fname != "figures.md":
            cloud_file = cloud_ref_dir / fname
            assert cloud_file.exists(), (
                f"Missing shared reference in cloud skill: {cloud_file}"
            )
            assert cloud_file.read_bytes() == shared_file.read_bytes(), (
                f"Byte mismatch in {fname} between skills/_shared/references and skills/shinystat-cloud/references"
            )


def test_src_medstat_and_skills_bundled_medstat_byte_identical():
    """Verify exact byte-identical parity between src/medstat (SSOT) and skills/shinystat/scripts/medstat."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src" / "medstat"
    assert src_dir.exists(), "src/medstat directory not found"

    targets = [
        (
            "canonical skills/",
            repo_root / "skills" / "shinystat" / "scripts" / "medstat",
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


def test_sync_skills_script_execution_and_drift_check():
    """Verify scripts/sync-skills.sh --check executes cleanly with zero git drift."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / ".git").exists():
        pytest.skip("Drift check requires a Git working tree")
    sync_script = repo_root / "scripts" / "sync-skills.sh"
    assert sync_script.exists(), "scripts/sync-skills.sh not found"

    res = subprocess.run(
        ["bash", str(sync_script), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, (
        f"sync-skills.sh --check failed:\n{res.stderr}\n{res.stdout}"
    )
    assert "Zero git drift verified in skills/." in res.stdout


def test_ci_drift_check_detects_untracked_mirror_files(tmp_path):
    """The CI drift command lists a mirror file that was never committed (ADR 69 Round 1)."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("Drift command needs git")
    repo_root = Path(__file__).resolve().parent.parent.parent
    workflow = yaml.safe_load(
        (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["verify-and-test"]["steps"]
    script = next(
        s["run"] for s in steps if s.get("name", "").startswith("Check SSOT sync")
    )
    # `git diff` does not show untracked files, so it cannot be the drift test.
    assert "git diff --quiet skills/" not in script
    assert "git diff --exit-code skills/" not in script
    match = re.search(r'^\s*DRIFT="\$\((?P<cmd>[^\n]+)\)"\s*$', script, re.MULTILINE)
    assert match, 'the drift step must capture drift as DRIFT="$(...)"'

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    def drift() -> str:
        res = subprocess.run(
            ["bash", "-c", match["cmd"]],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    tracked = tmp_path / "skills" / "tracked.md"
    tracked.parent.mkdir()
    tracked.write_text("a\n", encoding="utf-8")
    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "init",
    )
    assert drift() == ""

    new_mirror = tmp_path / "skills" / "new_module.py"
    new_mirror.write_text("x = 1\n", encoding="utf-8")
    assert "skills/new_module.py" in drift()
    new_mirror.unlink()

    tracked.write_text("b\n", encoding="utf-8")
    assert "skills/tracked.md" in drift()
    tracked.unlink()
    assert "skills/tracked.md" in drift()


def test_ci_workflow_keeps_event_data_out_of_shell_and_pins_actions():
    """Scripts read event data from env; actions are pinned to commits (ADR 69 Round 2)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    workflow = yaml.safe_load(
        (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    interpolated, unpinned = [], []
    for job_name, job in workflow["jobs"].items():
        for step in job["steps"]:
            # An expression inside `run` is pasted into the script before the shell
            # parses it, so a branch name could carry shell syntax.
            if "${{" in step.get("run", ""):
                interpolated.append(f"{job_name} / {step.get('name')}")
            uses = step.get("uses")
            if uses and not re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", uses):
                unpinned.append(uses)
    assert not interpolated, f"expressions inside run scripts: {interpolated}"
    assert not unpinned, f"actions not pinned to a commit SHA: {unpinned}"

    checkout = next(
        s
        for s in workflow["jobs"]["verify-and-test"]["steps"]
        if s.get("uses", "").startswith("actions/checkout@")
    )
    # A fork PR's head branch does not exist in this repository, so the head branch
    # is checked out only when the PR comes from this repository.
    # Same-repository PRs check out the head branch; fork PRs and non-PR events keep
    # the default ref.
    assert checkout["with"]["ref"] == (
        "${{ github.event.pull_request.head.repo.full_name == github.repository "
        "&& github.head_ref || github.ref }}"
    )


def test_engine_dependencies_equal_the_skill_runtime_packages():
    """pyproject declares the packages the bundled engine imports (ADR 69 Round 3)."""
    import tomllib

    repo_root = Path(__file__).resolve().parent.parent.parent
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!~ \[;]", dep, maxsplit=1)[0].lower()
        for dep in project["project"]["dependencies"]
    }
    requirements = repo_root / "skills" / "shinystat" / "requirements.txt"
    runtime = {
        line.partition("==")[0].strip().lower()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert declared == runtime, (
        f"only in pyproject: {sorted(declared - runtime)}; "
        f"only in requirements.txt: {sorted(runtime - declared)}"
    )


def test_design_md_adr_numbers_unique_and_sequential():
    """Verify DESIGN.md ADR headings are unique, sequential 1..68, and include ADR 68."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    design_path = repo_root / "DESIGN.md"
    assert design_path.exists(), "DESIGN.md not found at repository root"

    content = design_path.read_text(encoding="utf-8")
    adr_nums = [
        int(m) for m in re.findall(r"^## ADR (\d+):", content, flags=re.MULTILINE)
    ]
    assert adr_nums, "No ADR headings found in DESIGN.md"
    assert len(adr_nums) == len(set(adr_nums)), (
        f"Duplicate ADR numbers detected in DESIGN.md: {adr_nums}"
    )
    assert adr_nums == list(range(1, len(adr_nums) + 1)), (
        f"ADR numbers in DESIGN.md are not strictly sequential 1..N: {adr_nums}"
    )
    assert 68 in adr_nums and 69 in adr_nums, "Expected ADR 68 and 69 in DESIGN.md"


def test_subagent_protocol_and_gate5_delta_parity():
    """Verify Gate #5 Execution Context Mode delta and subagent-protocol.md sync across local and cloud skills."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_skill = repo_root / "skills" / "shinystat" / "SKILL.md"
    cloud_skill = repo_root / "skills" / "shinystat-cloud" / "SKILL.md"
    shared_proto = (
        repo_root / "skills" / "_shared" / "references" / "subagent-protocol.md"
    )

    assert shared_proto.exists(), (
        "skills/_shared/references/subagent-protocol.md missing"
    )
    proto_text = shared_proto.read_text(encoding="utf-8")

    # Ensure protocol defines required architectural invariants and zero medstat imports
    assert "from medstat" not in proto_text
    assert "import medstat" not in proto_text
    # Behavioral protocol clause and invariant verifications
    # 1. Unique run_id ownership & concurrency isolation
    assert ".scratch_checkpoint_<run_id>" in proto_text, (
        "Protocol must scope checkpoint directories with unique run_id"
    )
    assert (
        "verify `run_id` ownership before removing any staging artifacts" in proto_text
    ), "Protocol must require run_id ownership verification before cleanup"
    assert "table_dfs_<run_id>.json" in proto_text, (
        "Protocol must scope staging table artifacts with run_id"
    )
    assert "<run_output_dir>/table_dfs_<run_id>.json" in proto_text, (
        "Protocol must use run_output_dir for table assembly and cleanup"
    )
    assert "<run_output_dir>/table_dfs.json" in proto_text, (
        "Protocol must use run_output_dir for table_dfs.json"
    )
    assert "<run_output_dir>/results_dict.json" in proto_text, (
        "Protocol must use run_output_dir for results_dict.json"
    )
    assert "<output_dir>/results_dict.json" not in proto_text, (
        "Protocol must not use shared output_dir for results_dict.json"
    )
    assert "<run_output_dir>/sap_handoff.json" in proto_text, (
        "Protocol must use run_output_dir for sap_handoff.json"
    )
    assert "<output_dir>/sap_handoff.json" not in proto_text, (
        "Protocol must not use shared output_dir for sap_handoff.json"
    )
    assert "<run_output_dir>/.scratch_checkpoint_<run_id>/" in proto_text, (
        "Protocol must place checkpoints under run_output_dir"
    )
    assert "<output_dir>/.scratch_checkpoint_<run_id>/" not in proto_text, (
        "Protocol must not place checkpoints directly under shared output_dir"
    )
    assert "**outside** the user-facing `<output_dir>`" in proto_text, (
        "Protocol must require row-level checkpoints to be outside output_dir"
    )
    assert "cleaned_cohort.parquet` (or `.csv`) outside `<output_dir>`" in proto_text, (
        "Protocol must isolate row-level cleaned cohort files outside output_dir"
    )
    assert "never delete unowned or shared staging files" in proto_text, (
        "Protocol must forbid deleting unowned or shared staging files from other runs"
    )
    assert (
        "shared files outside `run_output_dir` are never deleted without verified ownership"
        in proto_text
    ), (
        "Protocol must protect shared files outside run_output_dir from unverified deletion"
    )
    assert (
        "persist the exact returned directory path (`row_level_checkpoint_dir`) and `run_id` in a role-specific file in the control directory"
        in proto_text
    ), (
        "Protocol must persist row_level_checkpoint_dir in control directory before writing cohort data"
    )
    assert (
        "reuse the registered workspace on later halts after verifying `run_id` ownership"
        in proto_text
    ), (
        "Protocol must reuse registered workspace across repeated halts for the same role and run"
    )
    assert (
        "Read the role-specific workspace registration record in the control directory to identify and remove the workspace even if Subagent B crashes before writing checkpoint state"
        in proto_text
    ), (
        "Protocol must read workspace registration during cleanup if Subagent B crashes before writing checkpoint state"
    )

    # 2. Finalized narrative citation re-verification
    assert "Finalized Narrative Citation Verification" in proto_text, (
        "Protocol must define finalized narrative citation verification"
    )
    assert (
        "reruns verification against the finalized Discussion and narrative text"
        in proto_text
    ), "Protocol must verify citations against finalized narrative text before delivery"
    assert (
        "when Subagent A was not engaged, compilation and delivery proceed without citation input"
        in proto_text
    ), (
        "Protocol must preserve that citation input is used only when Subagent A was engaged"
    )

    # 3. Preprocessing provenance invalidation & Gate #6 contract
    assert "preprocessing_provenance" in proto_text, (
        "Protocol must define preprocessing_provenance in checkpoint metadata"
    )
    assert "missing_data_strategy" in proto_text, (
        "Protocol must track missing_data_strategy in preprocessing provenance"
    )
    assert (
        "Invalidate `cleaned_data_path`, rerun cohort cleaning and sample retention tracking"
        in proto_text
    ), (
        "Protocol must invalidate cleaned_data_path when Gate #4 alters missingness strategy"
    )
    assert (
        "preserve the selective-resume behavior and load `cleaned_data_path` directly"
        in proto_text
    ), "Protocol must preserve selective resume when preprocessing remains compatible"
    assert (
        "Subagent A is required to read and apply `citation_instructions` when resuming the citation branch so it deterministically follows the user's latest Gate #6 decision"
        in proto_text
    ), (
        "Protocol must bind Subagent A to versioned citation_instructions on Gate #6 resume"
    )

    # 4. Independent Table-to-Results Numerical Reconciliation Audit & HALT on mismatch
    assert (
        "Independent Table-to-Results Numerical Reconciliation Audit" in proto_text
    ), "Protocol must enforce independent table-to-results reconciliation audit"
    assert "reconciling every numerical value displayed in table cells" in proto_text, (
        "Protocol must reconcile every numerical value in table cells against results_dict"
    )
    assert (
        "If this audit detects any mismatch—including altered odds ratios, transposed CI limits, or divergent counts—delivery must HALT immediately"
        in proto_text
    ), "Protocol must mandate immediate delivery HALT on any numerical mismatch"
    assert "does **NOT** substitute for the clinician's manual review" in proto_text, (
        "Protocol must enforce that automated checks do not replace manual review"
    )
    assert "unprefixed patient names" in proto_text, (
        "Protocol must require manual review of unprefixed patient names"
    )

    # 5. Executable Statistical Invariant & Precision Tolerance Verification
    def _verify_half_unit_tolerance(val: float, truth: float, decimals: int) -> bool:
        return abs(val - truth) <= 0.5 * 10 ** (-decimals) + 1e-9

    # Valid rounding within precision tolerance
    assert _verify_half_unit_tolerance(1.45, 1.452, 2) is True  # OR 1.45 matches 1.452
    assert (
        _verify_half_unit_tolerance(0.03, 0.034, 2) is True
    )  # P-val 0.03 matches 0.034
    assert (
        _verify_half_unit_tolerance(2.1, 2.14, 1) is True
    )  # CI upper 2.1 matches 2.14

    # Deviations exceeding half-unit precision tolerance must be rejected
    assert _verify_half_unit_tolerance(1.40, 1.452, 2) is False  # Altered OR rejected
    assert _verify_half_unit_tolerance(1.50, 1.452, 2) is False  # Inflated OR rejected
    assert (
        _verify_half_unit_tolerance(1.85, 2.14, 2) is False
    )  # Transposed CI bound rejected

    # Preprocessing provenance invalidation behavior test
    def _should_invalidate_cleaned_data(provenance: dict, gate4_decision: dict) -> bool:
        # Compare all recorded preprocessing settings determining cleaned-cohort compatibility
        keys_to_compare = (
            "missing_data_strategy",
            "imputation_m",
            "exclusion_criteria",
        )
        for k in keys_to_compare:
            if k in gate4_decision and provenance.get(k) != gate4_decision.get(k):
                return True
        return False

    prov = {
        "missing_data_strategy": "complete_case",
        "imputation_m": 0,
        "exclusion_criteria": "missing_lactate_dropped",
    }
    # Strategy changed to MICE -> must invalidate cleaned cohort
    assert (
        _should_invalidate_cleaned_data(
            prov,
            {
                "missing_data_strategy": "mice",
                "imputation_m": 10,
                "exclusion_criteria": "missing_lactate_dropped",
            },
        )
        is True
    )
    # Strategy unchanged and parameters identical -> selective resume preserved
    assert (
        _should_invalidate_cleaned_data(
            prov,
            {
                "missing_data_strategy": "complete_case",
                "imputation_m": 0,
                "exclusion_criteria": "missing_lactate_dropped",
            },
        )
        is False
    )
    # Same strategy (MICE) but parameter changed (e.g. imputation_m 10 -> 20) -> must invalidate
    mice_prov = {
        "missing_data_strategy": "mice",
        "imputation_m": 10,
        "exclusion_criteria": "none",
    }
    assert (
        _should_invalidate_cleaned_data(
            mice_prov,
            {
                "missing_data_strategy": "mice",
                "imputation_m": 20,
                "exclusion_criteria": "none",
            },
        )
        is True
    )
    # Same strategy but exclusion criteria changed -> must invalidate
    assert (
        _should_invalidate_cleaned_data(
            prov,
            {
                "missing_data_strategy": "complete_case",
                "imputation_m": 0,
                "exclusion_criteria": "missing_sofa_dropped",
            },
        )
        is True
    )

    for skill_path, expected_link in (
        (local_skill, "references/subagent-protocol.md"),
        (cloud_skill, "references/subagent-protocol.md"),
    ):
        text = skill_path.read_text(encoding="utf-8")
        assert "Execution Context Mode" in text, (
            f"Missing 'Execution Context Mode' in {skill_path}"
        )
        assert "Subagent Offload" in text, f"Missing 'Subagent Offload' in {skill_path}"
        assert "Direct in Main Agent" in text, (
            f"Missing 'Direct in Main Agent' default in {skill_path}"
        )
        assert "single consolidated turn" in text, (
            f"Missing 'single consolidated turn' pre-flight rule in {skill_path}"
        )
        assert "clinically plausible MNAR without sensitivity analysis" in text, (
            f"Missing pre-flight MNAR condition in {skill_path}"
        )
        assert "computing EPV after `#1`/`#2` resolution" in text, (
            f"Missing deferred EPV calculation rule for unspecified outcome in {skill_path}"
        )
        assert ".scratch_checkpoint/" in text, (
            f"Missing '.scratch_checkpoint/' completion criteria in {skill_path}"
        )
        assert expected_link in text, f"Missing link '{expected_link}' in {skill_path}"

    # Ensure install/package scripts sync subagent-protocol.md before checking packaged outputs
    package_script = repo_root / "scripts" / "package-cloud-skill.sh"
    if package_script.exists():
        subprocess.run(
            ["bash", str(package_script)],
            check=True,
            cwd=repo_root,
            capture_output=True,
        )

    local_proto = (
        repo_root / "skills" / "shinystat" / "references" / "subagent-protocol.md"
    )
    cloud_proto = (
        repo_root / "skills" / "shinystat-cloud" / "references" / "subagent-protocol.md"
    )
    packaged_proto = (
        repo_root
        / "packaging"
        / "cloud"
        / "shinystat-cloud"
        / "references"
        / "subagent-protocol.md"
    )
    assert (
        local_proto.exists() and local_proto.read_bytes() == shared_proto.read_bytes()
    )
    assert (
        cloud_proto.exists() and cloud_proto.read_bytes() == shared_proto.read_bytes()
    )
    if packaged_proto.exists():
        assert packaged_proto.read_bytes() == shared_proto.read_bytes()


def test_extract_heading_slugs_code_fences_and_links():
    """Verify _extract_heading_slugs ignores headings inside code fences and normalizes links."""
    doc = """# Normal Heading
Some text.
```bash
# This is a bash comment
cat << 'EOF'
## Fake Heading
EOF
```
~~~markdown
# Another fake heading
~~~
## [Link Heading](https://example.com/section)
### Subheading with [Doc Link](docs/guide.md#anchor) and `code`
"""
    slugs = _extract_heading_slugs(doc)
    assert "normal-heading" in slugs
    assert "link-heading" in slugs
    assert "subheading-with-doc-link-and-code" in slugs
    assert "this-is-a-bash-comment" not in slugs
    assert "fake-heading" not in slugs
    assert "another-fake-heading" not in slugs
    assert not any("examplecom" in s for s in slugs)
    assert not any("guide" in s for s in slugs)


def test_consort_flow_rendered_labels_single_field_inputs(monkeypatch, tmp_path):
    """Regression test against canonical skills/shinystat/references/flows.md.

    Verifies that when single-field inputs are supplied (e.g. allocated=100, received=80,
    analyzed=80), the derived missing counts (did_not_receive=20, excluded_from_analysis=20)
    are rendered as labels in the resulting diagram ('Did not receive intervention (n = 20)'
    and 'Excluded from analysis (n = 20)').
    """
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parent.parent.parent
    flows_file = repo_root / "skills" / "shinystat" / "references" / "flows.md"
    assert flows_file.exists(), "skills/shinystat/references/flows.md not found"

    flows_md = flows_file.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", flows_md, re.DOTALL)
    ns = {}
    for idx, b in enumerate(blocks):
        block_ns = {}
        try:
            exec(b, block_ns)
            ns.update(block_ns)
        except Exception as exc:
            pytest.fail(f"flows.md python block {idx} failed to execute: {exc!r}")

    draw_consort = ns["draw_standalone_consort_rct"]

    captured_texts = []
    orig_subplots = plt.subplots

    def wrap_subplots(*args, **kwargs):
        fig, ax = orig_subplots(*args, **kwargs)
        orig_savefig = fig.savefig

        def wrap_savefig(*sargs, **skwargs):
            for a in fig.axes:
                captured_texts.extend([t.get_text() for t in a.texts])
            return orig_savefig(*sargs, **skwargs)

        fig.savefig = wrap_savefig
        return fig, ax

    monkeypatch.setattr(plt, "subplots", wrap_subplots)
    out_png = tmp_path / "consort_rendered_labels.png"
    arms = [
        {
            "name": "Intervention",
            "allocated": 100,
            "received": 80,
            "analyzed": 80,
        },
        {
            "name": "Control",
            "allocated": 100,
            "did_not_receive": 0,
            "excluded_from_analysis": 0,
        },
    ]
    res = draw_consort(
        assessed=220,
        excluded={"Ineligible": 20},
        arms=arms,
        out_path=str(out_png),
    )
    assert Path(res).exists()

    all_rendered = "\n".join(captured_texts)
    assert "Did not receive intervention (n = 20)" in all_rendered, (
        "Derived 'Did not receive intervention (n = 20)' label not found in rendered text"
    )
    assert "Excluded from analysis (n = 20)" in all_rendered, (
        "Derived 'Excluded from analysis (n = 20)' label not found in rendered text"
    )
    assert "Lost to follow-up (unknown)" in all_rendered, (
        "Omitted lost count must be rendered as unknown"
    )
    assert "Discontinued intervention (unknown)" in all_rendered, (
        "Omitted discontinued count must be rendered as unknown"
    )

    # Explicit counts (including explicit 0) are rendered as (n = X)
    captured_texts.clear()
    out_png_explicit = tmp_path / "consort_explicit.png"
    draw_consort(
        assessed=220,
        excluded={"Ineligible": 20},
        arms=[
            {
                "name": "Intervention",
                "allocated": 100,
                "received": 80,
                "lost": 0,
                "discontinued": 5,
                "analyzed": 80,
            },
            {
                "name": "Control",
                "allocated": 100,
                "did_not_receive": 0,
                "lost": {"Relocated": 2},
                "discontinued": 0,
                "excluded_from_analysis": 0,
            },
        ],
        out_path=str(out_png_explicit),
    )
    all_explicit = "\n".join(captured_texts)
    assert "Lost to follow-up (n = 0)" in all_explicit
    assert "Discontinued intervention (n = 5)" in all_explicit
    assert "Lost to follow-up (n = 2)" in all_explicit
    assert "Discontinued intervention (n = 0)" in all_explicit


def test_consort_flow_count_normalization_and_unsigned_prevention(tmp_path):
    """Verifies that draw_standalone_consort_rct normalizes integral scalars to Python int
    and prevents unsigned arithmetic wrapping (e.g. np.uint64(50) - np.uint64(60)).
    """
    import numpy as np

    repo_root = Path(__file__).resolve().parent.parent.parent
    flows_file = repo_root / "skills" / "shinystat" / "references" / "flows.md"
    flows_md = flows_file.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", flows_md, re.DOTALL)
    ns = {}
    for idx, b in enumerate(blocks):
        block_ns = {}
        try:
            exec(b, block_ns)
            ns.update(block_ns)
        except Exception as exc:
            pytest.fail(f"flows.md block {idx} failed to execute: {exc!r}")

    draw_consort = ns["draw_standalone_consort_rct"]

    # 1. Unsigned integer subtraction cannot wrap: allocated=50, dnr=60 must raise ValueError
    with pytest.raises(
        ValueError, match="Follow-up and allocation counts must be non-negative"
    ):
        draw_consort(
            assessed=np.uint64(100),
            excluded={"Ineligible": np.uint64(0)},
            arms=[
                {
                    "name": "Arm A",
                    "allocated": np.uint64(50),
                    "did_not_receive": np.uint64(60),
                    "analyzed": np.uint64(50),
                },
                {
                    "name": "Arm B",
                    "allocated": np.uint64(50),
                    "received": np.uint64(50),
                    "analyzed": np.uint64(50),
                },
            ],
            out_path=str(tmp_path / "uint_wrap.png"),
        )

    # 2. NumPy integer scalars are successfully normalized and accepted
    out_png = tmp_path / "numpy_scalars.png"
    res = draw_consort(
        assessed=np.int64(200),
        excluded={"Ineligible": np.int64(20)},
        arms=[
            {
                "name": "Arm A",
                "allocated": np.int64(90),
                "received": np.int64(90),
                "analyzed": np.int64(90),
            },
            {
                "name": "Arm B",
                "allocated": np.int64(90),
                "received": np.int64(90),
                "analyzed": np.int64(90),
            },
        ],
        out_path=str(out_png),
    )
    assert Path(res).exists()


def test_house_styles_range_index_preservation(tmp_path):
    """Verifies that save_standalone_bundle preserves non-default RangeIndex coordinates
    in _source.csv while omitting default 0-indexed RangeIndex unless explicitly selected.
    """
    import pandas as pd

    repo_root = Path(__file__).resolve().parent.parent.parent
    hs_file = repo_root / "skills" / "shinystat" / "references" / "house-styles.md"
    hs_md = hs_file.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", hs_md, re.DOTALL)
    ns = {}
    for idx, b in enumerate(blocks):
        block_ns = {}
        try:
            exec(b, block_ns)
            ns.update(block_ns)
        except Exception as exc:
            pytest.fail(f"house-styles.md block {idx} failed to execute: {exc!r}")

    save_bundle = ns["save_standalone_bundle"]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 20, 30])

    try:
        # Non-default unnamed RangeIndex(1, 4): must be included in CSV
        df_non_default = pd.DataFrame({"y": [10, 20, 30]}, index=pd.RangeIndex(1, 4))
        res_nd = save_bundle(
            fig, str(tmp_path / "fig_nd.png"), source_df=df_non_default
        )
        csv_nd = Path(res_nd["csv_path"]).read_text(encoding="utf-8")
        assert "1,10" in csv_nd or ",y\n1,10" in csv_nd

        # Default unnamed RangeIndex(0, 3): must be omitted by default
        df_default = pd.DataFrame({"y": [10, 20, 30]}, index=pd.RangeIndex(0, 3))
        res_def = save_bundle(fig, str(tmp_path / "fig_def.png"), source_df=df_default)
        csv_def = Path(res_def["csv_path"]).read_text(encoding="utf-8")
        assert csv_def.splitlines()[0].strip() == "y"

        # Default unnamed RangeIndex(0, 3) with include_index=True: must be included
        res_exp = save_bundle(
            fig, str(tmp_path / "fig_exp.png"), source_df=df_default, include_index=True
        )
        csv_exp = Path(res_exp["csv_path"]).read_text(encoding="utf-8")
        assert "0,10" in csv_exp or ",y\n0,10" in csv_exp
    finally:
        plt.close(fig)


def test_gate5_asks_each_output_axis_separately():
    """Gate #5 asks format, house style, polish, and execution context as separate questions."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    for skill_path in (
        repo_root / "skills" / "shinystat" / "SKILL.md",
        repo_root / "skills" / "shinystat-cloud" / "SKILL.md",
    ):
        text = skill_path.read_text(encoding="utf-8")
        assert "**Output Alignment Gate**" in text, skill_path
        assert "one separate question per axis" in text, skill_path
        for axis in (
            "5a. Deliverable Format",
            "5b. Figure House Style",
            "5c. Manuscript Polish Mode",
            "5d. Execution Context Mode",
        ):
            assert axis in text, f"Missing Gate #5 axis '{axis}' in {skill_path}"
        # Every house style is its own option (no merged `CLINICAL` / `OKABE_ITO`)
        assert "`CLINICAL` / `OKABE_ITO`" not in text, skill_path
        # Execution context is asked whenever a subagent tool exists, independent of other axes
        assert "even when every other axis is specified" in text, skill_path


def test_skill_process_audit_f1_to_f7_and_q3():
    """Verify F1-F7 process improvements and Q3 local/cloud report-builder separation."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_skill_path = repo_root / "skills" / "shinystat" / "SKILL.md"
    cloud_skill_path = repo_root / "skills" / "shinystat-cloud" / "SKILL.md"
    local_text = local_skill_path.read_text(encoding="utf-8")
    cloud_text = cloud_skill_path.read_text(encoding="utf-8")

    # F1: Explicit Run Sequence with 6 steps and step-specific 'Done when' completion criteria
    for text, label in ((local_text, "local"), (cloud_text, "cloud")):
        assert "## 0. Run Sequence" in text, f"Missing Run Sequence in {label}"
        run_seq_match = re.search(
            r"## 0\. Run Sequence\n\n(.*?)(?=\n---\n|\n## |\Z)", text, re.DOTALL
        )
        assert run_seq_match, f"Failed to extract Run Sequence section in {label}"
        run_seq_text = run_seq_match.group(1)

        step_names = ["Inspect", "Ask", "Plan", "Execute", "Render", "Audit"]
        raw_steps = re.findall(
            r"(?:^|\n)\d+\.[ \t]+(?:\*\*([^\n*]+)\*\*|([^\n:.]+?)(?::|\.|\s*\n|$))",
            run_seq_text,
        )
        extracted_steps = [(m[0] or m[1]).strip() for m in raw_steps]
        assert extracted_steps == step_names, (
            f"Run Sequence in {label} must match exact ordered steps without duplicates or extra steps: expected {step_names}, got {extracted_steps}"
        )

        last_step_pos = -1
        for idx, step_name in enumerate(step_names, 1):
            step_pattern = rf"(?:^|\n){idx}\.[ \t]+\*\*{step_name}\*\*(.*?)(?=\n\d+\.[ \t]+\*\*|\Z)"
            step_match = re.search(step_pattern, run_seq_text, re.DOTALL)
            assert step_match, (
                f"Missing step {idx}. **{step_name}** in {label} Run Sequence"
            )
            assert step_match.start() > last_step_pos, (
                f"Step {idx}. **{step_name}** in {label} Run Sequence is out of order"
            )
            last_step_pos = step_match.start()
            step_body = step_match.group(1)
            assert "*Done when*" in step_body, (
                f"Step {idx}. **{step_name}** in {label} must contain its own '*Done when*' criterion"
            )
            criterion = step_body.split("*Done when*", 1)[1].strip()
            assert len(criterion) > 0, (
                f"Empty '*Done when*' criterion for {step_name} in {label}"
            )
            if step_name == "Execute":
                crit_lower = criterion.lower()
                assert "converge" in crit_lower and "diagnostic" in crit_lower, (
                    f"Execute step in {label} must retain convergence and diagnostics requirement, got: {criterion}"
                )

    # F2: Disclosed Autonomous Upgrades into decision-heuristics.md (reduced SKILL.md size)
    assert len(local_text.encode("utf-8")) <= 16500, (
        f"Local SKILL.md still too large: {len(local_text.encode('utf-8'))} bytes"
    )
    assert len(cloud_text.encode("utf-8")) <= 22000, (
        f"Cloud SKILL.md still too large: {len(cloud_text.encode('utf-8'))} bytes"
    )

    # F3: Distinct front-loaded descriptions for local vs cloud
    local_fm = yaml.safe_load(local_text.split("---", 2)[1])
    cloud_fm = yaml.safe_load(cloud_text.split("---", 2)[1])
    assert local_fm["description"].startswith("Local clinical biostatistics"), local_fm[
        "description"
    ]
    assert cloud_fm["description"].startswith("Cloud-sandbox clinical biostatistics"), (
        cloud_fm["description"]
    )

    # F5: README deduplicates the 24-row Recipe Index table
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "| **Recipe 18** |" not in readme_text, (
        "Duplicate 24-row recipe table should be linked from README.md, not duplicated"
    )

    # F6: Multiple-choice tool instruction at Gate #2 / Gate #5
    for text, label in ((local_text, "local"), (cloud_text, "cloud")):
        assert "ask_question" in text, f"Missing ask_question guidance in {label}"
        assert re.search(
            r"when your tools include `?ask_question`?, send the consolidated turn through it",
            text,
            re.IGNORECASE,
        ), (
            f"Missing directive to route consolidated inquiry through ask_question in {label}"
        )

    # F7 & Q3: Dedicated evidence-pipeline.md & local medstat.reporting renderers
    shared_ep = repo_root / "skills" / "_shared" / "references" / "evidence-pipeline.md"
    assert shared_ep.exists(), "Missing skills/_shared/references/evidence-pipeline.md"
    assert "Step 1: Clinical Claims vs Analyzed Data Audit" in shared_ep.read_text(
        encoding="utf-8"
    )
    assert "references/evidence-pipeline.md" in local_text
    assert "references/evidence-pipeline.md" in cloud_text
    assert "medstat.reporting.renderers" in local_text
    assert "render_html" in local_text and "render_docx" in local_text


def test_live_eval_wording_rules_adr80():
    """ADR 80: gate and reporting rules that live-agent runs applied inconsistently."""
    root = Path(__file__).resolve().parents[2] / "skills"
    local = (root / "shinystat" / "SKILL.md").read_text(encoding="utf-8")
    cloud = (root / "shinystat-cloud" / "SKILL.md").read_text(encoding="utf-8")
    heur = (root / "_shared" / "references" / "decision-heuristics.md").read_text(encoding="utf-8")
    for text in (local, cloud):
        assert "≥2 distinct estimands (a primary model and its sensitivity re-fits count as one)" in text  # 5d trigger
        assert "biomarker direction not fixed by" in text  # Gate #1
        assert "primary, Complete-Case sensitivity" in text  # MICE triage
        assert "**Effect Scale**" in text
    # ADR 81: audit the delivered reply and exported data files; a figure per primary estimate.
    assert "on that same `ReportDocument` with `exported_files`" in local
    assert "`delivered_text` (In-Chat Summary reply)" in local
    assert "(In-Chat Summary reply) with every data file written as `exported_files`" in cloud
    assert "each primary estimate is plotted" in local and "each primary estimate is plotted" in cloud
    assert "**Identifier-Free Exports**" in cloud
    for text in (local, cloud):
        assert "spline df; no intercept)" in text
    assert "GEE logistic clustered on pair ID" in heur
    assert "`medstat.models.firth.check_separation` on every logistic design (outcome and propensity models)" in local  # Gate #3 detector
    assert "`plot_dual_*` figures" in local  # 5a dual view
    assert "pandas/scipy" not in cloud.split("---")[1]  # description names no library subset (ADR 81 Round 1)
    assert "fit_iptw_outcome_model" in heur and "fit_weighted_outcome_model" in heur
    assert "MICE is the primary analysis" in heur and "the retention flow ends at the primary analytic N" in heur


def test_api_quickref_matches_engine():
    """Every signature line in api-quickref.md equals inspect.signature of the engine (ADR 80)."""
    import importlib
    import inspect

    sheet = (
        Path(__file__).resolve().parents[2] / "skills" / "shinystat" / "references" / "api-quickref.md"
    ).read_text(encoding="utf-8")
    block = re.search(r"```text\n(.*?)\n```", sheet, re.DOTALL).group(1)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    assert len(lines) >= 10

    def _resolve(qual: str):
        parts = qual.split(".")
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
                break
            except ModuleNotFoundError:
                continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj

    for ln in lines:
        qual = ln.split("(", 1)[0]
        sig = inspect.signature(_resolve(qual))
        params = []
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            if p.kind is p.VAR_KEYWORD:
                params.append(f"**{p.name}")
                continue
            params.append(p.name if p.default is p.empty else f"{p.name}={p.default!r}")
        ret = sig.return_annotation
        ret_s = "" if ret is sig.empty or ret in ("None", None) else f" -> {ret}"
        assert ln == f"{qual}({', '.join(params)}){ret_s}", f"Stale quick-reference line: {ln}"


def test_live_eval_round_two_wording_and_engine_adr85():
    """ADR 85: fixes from 26 live-agent replicates (Thai fonts, retention QA, wording, quick reference)."""
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    from medstat.figures.journal import apply_journal_theme, thai_fallback_family
    from medstat.figures.retention import plot_retention_flow

    root = Path(__file__).resolve().parents[2] / "skills"
    local = (root / "shinystat" / "SKILL.md").read_text(encoding="utf-8")
    cloud = (root / "shinystat-cloud" / "SKILL.md").read_text(encoding="utf-8")
    heur = (root / "_shared" / "references" / "decision-heuristics.md").read_text(encoding="utf-8")
    flows = (root / "_shared" / "references" / "flows.md").read_text(encoding="utf-8")
    quick = (root / "shinystat" / "references" / "api-quickref.md").read_text(encoding="utf-8")

    # Thai labels render in every journal theme when a Thai face is installed.
    installed = {f.name for f in fm.fontManager.ttflist}
    if installed & {"Sarabun", "Thonburi", "Sukhumvit Set", "Ayuthaya", "Tahoma", "Arial Unicode MS"}:
        apply_journal_theme("NEJM")
        assert len(matplotlib.rcParams["font.family"]) >= 2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            ax.set_title("มาตรฐาน vs ทดลอง")
            fig.canvas.draw()
            plt.close(fig)
        assert not [w for w in caught if "missing from font" in str(w.message)]
    assert thai_fallback_family(["No Such Face"])[-1] in {"DejaVu Sans", "sans-serif"}

    # plot_retention_flow reports layout QA like every other figure.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        res = plot_retention_flow(
            {"n_initial": 100, "n_excluded": 10, "n_analyzed": 90, "exclusion_reasons": ["Missing outcome"]},
            out_path=Path(tmp) / "flow.png",
        )
        assert res.qa is not None and res.qa["passed"] is True
        assert res.source_df["Count"].tolist() == [100, 10, 90]

    # Run Sequence: separation and Little's test run at pre-flight; Gate #3 joins the consolidated ask.
    assert "`check_separation` on each dummy-coded logistic design" in local
    assert "`#1`–`#4`" in local and "`#1`–`#4`" in cloud
    for text in (local, cloud):
        assert "chat message" in text  # chat reply audited for file deliverables too
    assert "render_pptx" in local and "read only for the PDF fallback" in local

    # Heuristics: covariate >40% missing halts, missing outcome asks, ATT vs ATO, units, effect scale.
    assert "any model covariate counts as non-primary" in heur
    assert "ask at Gate #1 unless the user gave a disposition" in heur
    assert "never switch silently" in heur
    assert "per 10 years (age, when the data record it in years) or per 1 SD" in heur
    assert "`EffectScaleError`" in heur
    assert "never derive `received` from `allocated` alone" in flows
    assert "deliver those, labelled partial" in flows
    assert "without observed receipt/analysis counts" in cloud
    for text in (local, cloud):
        assert "in the exposure or any covariate (Little's test cannot lift it)" in text
        assert "`#2`–`#4` first found mid-run" in text
    assert "Primary Exposure Missing > 40%" in heur and "the exposure is never dropped" in heur
    assert "when the data record it in years" in heur
    assert "(rows complete on its variables)" in local
    assert "(rows complete on its variables) per §2.B" in local
    assert "when its outcome holds one class (report the absent class)" in heur
    assert "name the aliased columns from the error" in heur
    for text in (local, cloud):
        assert "name tested variables (untested or df 0 ≠ MCAR)" in text
    assert "Primary Exposure Missing = 100% in the analytic cohort" in heur
    assert "a categorical or otherwise excluded variable as untested, never as MCAR" in heur
    assert "Mark separation unevaluable and continue to the gate questions" in heur
    assert "after applying the approved outcome disposition" in heur
    assert "the local run halts and reports the exposure as not locally imputable" in heur
    assert "is reported as unavailable, never as MCAR" in heur
    assert "`check_separation` on each dummy-coded logistic design" in local
    assert "compute that SD once from the observed values" in heur
    assert "the primary analysis is high-M MICE (M ≥ 40, tracking FMI) that imputes the exposure" in heur
    assert "without observed receipt/analysis counts" in local
    assert "for an In-Chat Summary render nothing" in quick
    assert "pooled effect scale" in local and "pooled effect scale" in cloud
    assert "No estimand named by the user" in heur
    assert "do not change the primary model silently" in heur
    assert "medstat.causal.psm.propensity_score_match(" in quick
    assert "`FigureResult.csv_path`" in quick and "rename the strata columns" in quick

    # Quick reference now lists the functions every replicate had to look up in engine source.
    for needle in (
        "medstat.diagnostic.roc.auc_ci_delong(",
        "medstat.models.survival.fit_cox_ph(",
        "medstat.reporting.ir.ReportDocument.add_table(",
        "medstat.reporting.renderers.render_pptx(",
        "medstat.agreement.icc.calculate_icc(",
        "medstat.data.missing.pool_regression_results(",
    ):
        assert needle in quick, needle
