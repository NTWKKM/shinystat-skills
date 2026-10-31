#!/usr/bin/env bash
# scripts/install-skills.sh
# Single-Source-of-Truth installer for shinystat-skills (Phase 1 & Phase 3)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="--symlink"
SCOPE="workspace"
TARGET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --copy)
            MODE="--copy"
            shift
            ;;
        --symlink)
            MODE="--symlink"
            shift
            ;;
        --global)
            SCOPE="global"
            shift
            ;;
        -s|--scope)
            SCOPE="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

link_or_copy() {
    local src="$1"
    local dst="$2"
    rm -rf "$dst"
    mkdir -p "$(dirname "$dst")"
    if [[ "$MODE" == "--copy" ]]; then
        cp -R "$src" "$dst"
    else
        # ใช้ relative symlink เพื่อให้ย้ายโฟลเดอร์โปรเจกต์ได้โดยลิงก์ไม่ขาด
        python3 -c "import os, sys; os.symlink(os.path.relpath(sys.argv[1], os.path.dirname(sys.argv[2])), sys.argv[2])" "$src" "$dst"
    fi
}

echo "[1/3] Syncing shared references into shinystat and shinystat-cloud..."
SHARED_REF_DIR="$REPO_ROOT/skills/_shared/references"
if [[ -d "$SHARED_REF_DIR" ]]; then
    for ref_file in "$SHARED_REF_DIR"/*.md; do
        [[ -e "$ref_file" ]] || continue
        fname="$(basename "$ref_file")"
        link_or_copy "$ref_file" "$REPO_ROOT/skills/shinystat/references/$fname"
        if [[ "$fname" != "figures.md" ]]; then
            link_or_copy "$ref_file" "$REPO_ROOT/skills/shinystat-cloud/references/$fname"
        fi
    done
fi

echo "[2/3] Linking canonical src/medstat into skills/shinystat/scripts/medstat..."
link_or_copy "$REPO_ROOT/src/medstat" "$REPO_ROOT/skills/shinystat/scripts/medstat"

echo "[3/3] Installing cloud skill into packaging/cloud..."
mkdir -p "$REPO_ROOT/packaging/cloud"
link_or_copy "$REPO_ROOT/skills/shinystat-cloud" "$REPO_ROOT/packaging/cloud/shinystat-cloud"
if [[ "$MODE" == "--copy" ]]; then
    link_or_copy "$REPO_ROOT/skills/_shared" "$REPO_ROOT/packaging/cloud/_shared"
else
    rm -rf "$REPO_ROOT/packaging/cloud/_shared"
fi

CANONICAL_AGENT="$REPO_ROOT/agents/figure-tester.md"

if [[ "$SCOPE" == "global" ]]; then
    echo "[Global] Installing standalone dereferenced skills into Antigravity global config..."
    GLOBAL_DIRS=()
    if [[ "$TARGET" == "all" || "$TARGET" == "antigravity" ]]; then
        GLOBAL_DIRS+=(
            "${HOME}/.gemini/config/skills"
            "${HOME}/.gemini/antigravity/skills"
        )
    fi
    if [[ "$TARGET" == "all" || "$TARGET" == "claude" ]]; then
        GLOBAL_DIRS+=("${HOME}/.claude/skills")
    fi
    if [[ "$TARGET" == "all" || "$TARGET" == "cursor" ]]; then
        GLOBAL_DIRS+=("${HOME}/.cursor/skills")
    fi

    # Ensure packaging/cloud is built if needed
    if [[ ! -d "$REPO_ROOT/packaging/cloud/shinystat-cloud" || -L "$REPO_ROOT/packaging/cloud/shinystat-cloud" ]]; then
        "$REPO_ROOT/scripts/package-cloud-skill.sh" >/dev/null
    fi

    for gdir in "${GLOBAL_DIRS[@]}"; do
        mkdir -p "$gdir"
        rm -rf "$gdir/shinystat" "$gdir/shinystat-cloud"
        cp -RL "$REPO_ROOT/skills/shinystat" "$gdir/shinystat"
        cp -RL "$REPO_ROOT/packaging/cloud/shinystat-cloud" "$gdir/shinystat-cloud"
        find "$gdir/shinystat" "$gdir/shinystat-cloud" -name ".DS_Store" -delete 2>/dev/null || true
        find "$gdir/shinystat" "$gdir/shinystat-cloud" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        echo "  ✓ Installed into $gdir"
    done

    # Global agent
    if [[ -f "$CANONICAL_AGENT" && -d "${HOME}/.gemini/config/agents" ]]; then
        cp "$CANONICAL_AGENT" "${HOME}/.gemini/config/agents/figure-tester.md"
        echo "  ✓ Installed figure-tester agent into ${HOME}/.gemini/config/agents/figure-tester.md"
    fi
fi

echo "Done! Installed shinystat-skills in '$MODE' mode (scope: $SCOPE)."
