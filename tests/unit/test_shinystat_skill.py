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
