#!/usr/bin/env bash
# ==============================================================================
# install-skills.sh — Agent Skills Installer for shinystat
# Installs atomic, self-contained Agent Skills into Antigravity, Claude Code,
# and Cursor environments.
# ==============================================================================

set -euo pipefail

CALLER_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILLS_SRC="${REPO_ROOT}/skills"

TARGET="all"
SCOPE="workspace"
CUSTOM_DEST=""
SYNC_ONLY=false

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install shinystat Agent Skills into your agent environment.

Options:
  -t, --target TARGET    Target platform: antigravity, claude, cursor, all
                         (Default: all)
  -s, --scope SCOPE      Installation scope: workspace, global
                         (Default: workspace)
  -d, --dest DIR         Custom destination directory (overrides target & scope)
  --sync, --sync-only    Synchronize src/medstat (SSOT) -> skills/ and update workspace mirrors
  -h, --help             Show this help message and exit

Examples:
  # Install into current workspace for all supported platforms
  ./scripts/install-skills.sh

  # Synchronize src/medstat into skills/ and all workspace agent mirrors
  ./scripts/install-skills.sh --sync

  # Install into Antigravity workspace (.agents/skills/)
  ./scripts/install-skills.sh --target antigravity --scope workspace

  # Install globally for Claude Code (~/.claude/skills/)
  ./scripts/install-skills.sh --target claude --scope global

  # Install into a custom agent skills directory
  ./scripts/install-skills.sh --dest /path/to/my/skills
EOF
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -s|--scope)
            SCOPE="$2"
            shift 2
            ;;
        --workspace)
            SCOPE="workspace"
            shift
            ;;
        --global)
            SCOPE="global"
            shift
            ;;
        --sync|--sync-only)
            SYNC_ONLY=true
            shift
            ;;
        -d|--dest)
            CUSTOM_DEST="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            print_usage
            exit 1
            ;;
    esac
done

# Synchronize src/medstat (Single Source of Truth) into skills/shinystat/scripts/medstat
MEDSTAT_SRC="${REPO_ROOT}/src/medstat"
MEDSTAT_BUNDLE="${SKILLS_SRC}/shinystat/scripts/medstat"

if [[ -d "${MEDSTAT_SRC}" && -d "${MEDSTAT_BUNDLE}" ]]; then
    echo "▶ Synchronizing src/medstat (SSOT) -> skills/shinystat/scripts/medstat..."
    rsync -a --delete --exclude="__pycache__" --exclude="*.pyc" "${MEDSTAT_SRC}/" "${MEDSTAT_BUNDLE}/"
fi

# Synchronize skills/shinystat-cloud -> packaging/cloud/shinystat-cloud
CLOUD_PKG="${REPO_ROOT}/packaging/cloud/shinystat-cloud"
CLOUD_SKILL="${SKILLS_SRC}/shinystat-cloud"
if [[ -d "${CLOUD_SKILL}" && -d "${CLOUD_PKG}" ]]; then
    rsync -a --delete --exclude="__pycache__" --exclude="*.pyc" "${CLOUD_SKILL}/" "${CLOUD_PKG}/"
fi

if [[ ! -d "${SKILLS_SRC}" ]]; then
    echo "Error: Source skills directory not found at '${SKILLS_SRC}'" >&2
    exit 1
fi

SKILL_DIRS=()
for s in "${SKILLS_SRC}"/*; do
    if [[ -d "$s" && -f "$s/SKILL.md" ]]; then
        SKILL_DIRS+=("$(basename "$s")")
    fi
done

if [[ ${#SKILL_DIRS[@]} -eq 0 ]]; then
    echo "Error: No valid skills found in '${SKILLS_SRC}'" >&2
    exit 1
fi

echo "=========================================================="
echo "  shinystat Agent Skills Installer"
echo "=========================================================="
echo "Found ${#SKILL_DIRS[@]} skills in ${SKILLS_SRC}:"
for s in "${SKILL_DIRS[@]}"; do
    echo "  - $s"
done
echo "=========================================================="

copy_skills() {
    local dest_dir="$1"
    local env_name="$2"

    echo ""
    echo "▶ Installing into ${env_name}: ${dest_dir}"
    mkdir -p "${dest_dir}"

    for skill in "${SKILL_DIRS[@]}"; do
        local src_path="${SKILLS_SRC}/${skill}"
        local target_path="${dest_dir}/${skill}"
        rm -rf "${target_path}"
        cp -R "${src_path}" "${target_path}"
        echo "  ✓ Installed ${skill} -> ${target_path}"
    done
}

if [[ "${SYNC_ONLY}" == true ]]; then
    echo ""
    echo "▶ Synchronizing workspace agent mirrors..."
    MIRRORS=(
        "${REPO_ROOT}/.agents/skills|Antigravity (.agents)"
        "${REPO_ROOT}/.agent/skills|Antigravity (.agent)"
        "${REPO_ROOT}/.claude/skills|Claude Code (.claude)"
        "${REPO_ROOT}/.cursor/skills|Cursor (.cursor)"
    )
    for entry in "${MIRRORS[@]}"; do
        IFS="|" read -r dir_path label <<< "${entry}"
        copy_skills "${dir_path}" "${label}"
    done

    # Synchronize packaging/cloud/shinystat-cloud for Claude Web packaging
    rm -rf "${REPO_ROOT}/packaging/cloud/shinystat-cloud"
    mkdir -p "${REPO_ROOT}/packaging/cloud"
    cp -R "${REPO_ROOT}/skills/shinystat-cloud" "${REPO_ROOT}/packaging/cloud/shinystat-cloud"
    echo "  ✓ Installed shinystat-cloud -> ${REPO_ROOT}/packaging/cloud/shinystat-cloud"

    echo ""
    echo "=========================================================="
    echo "  ✓ Synchronization complete! All mirrors match src/medstat."
    echo "=========================================================="
    exit 0
fi

if [[ -n "${CUSTOM_DEST}" ]]; then
    copy_skills "${CUSTOM_DEST}" "Custom Destination"
    echo ""
    echo "Installation complete!"
    exit 0
fi

# Determine destination directories
DESTINATIONS=()

case "${TARGET}" in
    antigravity)
        if [[ "${SCOPE}" == "global" ]]; then
            DESTINATIONS+=("${HOME}/.gemini/config/skills|Antigravity (Global config)")
            DESTINATIONS+=("${HOME}/.gemini/antigravity/skills|Antigravity (Global antigravity)")
        else
            DESTINATIONS+=("${CALLER_CWD}/.agents/skills|Antigravity (Workspace .agents)")
            DESTINATIONS+=("${CALLER_CWD}/.agent/skills|Antigravity (Workspace .agent singular)")
        fi
        ;;
    claude)
        if [[ "${SCOPE}" == "global" ]]; then
            DESTINATIONS+=("${HOME}/.claude/skills|Claude Code (Global)")
        else
            DESTINATIONS+=("${CALLER_CWD}/.claude/skills|Claude Code (Workspace)")
        fi
        ;;
    cursor)
        if [[ "${SCOPE}" == "global" ]]; then
            DESTINATIONS+=("${HOME}/.cursor/skills|Cursor (Global)")
        else
            DESTINATIONS+=("${CALLER_CWD}/.cursor/skills|Cursor (Workspace .cursor)")
            DESTINATIONS+=("${CALLER_CWD}/.agents/skills|Cursor (Workspace .agents)")
        fi
        ;;
    all)
        if [[ "${SCOPE}" == "global" ]]; then
            DESTINATIONS+=("${HOME}/.gemini/config/skills|Antigravity (Global config)")
            DESTINATIONS+=("${HOME}/.gemini/antigravity/skills|Antigravity (Global antigravity)")
            DESTINATIONS+=("${HOME}/.claude/skills|Claude Code (Global)")
            DESTINATIONS+=("${HOME}/.cursor/skills|Cursor (Global)")
        else
            DESTINATIONS+=("${CALLER_CWD}/.agents/skills|Antigravity (Workspace .agents)")
            DESTINATIONS+=("${CALLER_CWD}/.agent/skills|Antigravity (Workspace .agent singular)")
            DESTINATIONS+=("${CALLER_CWD}/.claude/skills|Claude Code (Workspace)")
            DESTINATIONS+=("${CALLER_CWD}/.cursor/skills|Cursor (Workspace .cursor)")
        fi
        ;;
    *)
        echo "Error: Invalid target '${TARGET}'. Allowed: antigravity, claude, cursor, all" >&2
        exit 1
        ;;
esac

for entry in "${DESTINATIONS[@]}"; do
    IFS="|" read -r dir_path label <<< "${entry}"
    copy_skills "${dir_path}" "${label}"
done

echo ""
echo "=========================================================="
echo "  All shinystat Agent Skills successfully installed!"
echo "=========================================================="
