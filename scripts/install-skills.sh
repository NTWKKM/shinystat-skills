#!/usr/bin/env bash
# scripts/install-skills.sh
# Single-Source-of-Truth installer for shinystat-skills (Phase 1 & Phase 3)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="--symlink"
SCOPE="workspace"
TARGET="all"
DRY_RUN=false

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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--scope)
            if [[ $# -lt 2 || "$2" =~ ^- ]]; then
                echo "Error: Missing or invalid argument for $1" >&2
                exit 1
            fi
            SCOPE="$2"
            shift 2
            ;;
        -t|--target)
            if [[ $# -lt 2 || "$2" =~ ^- ]]; then
                echo "Error: Missing or invalid argument for $1" >&2
                exit 1
            fi
            TARGET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--symlink|--copy] [--global] [-s|--scope workspace|global] [-t|--target all|antigravity|claude|cursor] [--dry-run]"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$SCOPE" != "workspace" && "$SCOPE" != "global" ]]; then
    echo "Error: Invalid scope '$SCOPE'. Must be one of: workspace, global" >&2
    exit 1
fi

if [[ "$TARGET" != "all" && "$TARGET" != "antigravity" && "$TARGET" != "claude" && "$TARGET" != "cursor" ]]; then
    echo "Error: Invalid target '$TARGET'. Must be one of: all, antigravity, claude, cursor" >&2
    exit 1
fi

link_or_copy() {
    local src="$1"
    local dst="$2"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] link_or_copy $src -> $dst"
        return 0
    fi
    rm -rf "$dst"
    mkdir -p "$(dirname "$dst")"
    if [[ "$MODE" == "--copy" ]]; then
        cp -R "$src" "$dst"
    else
        # ใช้ relative symlink เพื่อให้ย้ายโฟลเดอร์โปรเจกต์ได้โดยลิงก์ไม่ขาด
        python3 -c "import os, sys; os.symlink(os.path.relpath(sys.argv[1], os.path.dirname(sys.argv[2])), sys.argv[2])" "$src" "$dst"
    fi
}

remove_legacy_mirror() {
    local target_path="$1"
    [[ -e "$target_path" || -L "$target_path" ]] || return 0

    local is_mirror
    is_mirror=$(python3 - "$target_path" "$REPO_ROOT" << 'PY'
import sys, os

target = sys.argv[1]
repo_root = os.path.realpath(sys.argv[2])

# Check if target is a symlink
if os.path.islink(target):
    try:
        raw_dest = os.readlink(target)
        if not os.path.isabs(raw_dest):
            raw_dest = os.path.normpath(os.path.join(os.path.dirname(target), raw_dest))
        resolved_dest = os.path.realpath(raw_dest)
    except Exception:
        resolved_dest = ""

    skills_dir = os.path.realpath(os.path.join(repo_root, "skills"))
    pkg_dir = os.path.realpath(os.path.join(repo_root, "packaging"))

    if (
        resolved_dest.startswith(skills_dir + os.sep)
        or resolved_dest == skills_dir
        or resolved_dest.startswith(pkg_dir + os.sep)
        or resolved_dest == pkg_dir
        or resolved_dest.startswith(repo_root + os.sep)
    ):
        print("true")
        sys.exit(0)
    print("false")
    sys.exit(0)

# Non-symlink directory or file
norm_target = os.path.normpath(target)
bname = os.path.basename(norm_target)
pname = os.path.basename(os.path.dirname(norm_target))

if pname == "skills" and bname in ("shinystat", "shinystat-cloud", "_shared"):
    print("true")
    sys.exit(0)

if pname == "agents" and bname == "figure-tester.md":
    print("true")
    sys.exit(0)

print("false")
PY
)

    if [[ "$is_mirror" == "true" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] remove $target_path"
        else
            if [[ -d "$target_path" && ! -L "$target_path" ]]; then
                rm -rf "$target_path"
            else
                rm -f "$target_path"
            fi
            echo "[remove] $target_path"
        fi
    else
        echo "[skip] $target_path (not managed by shinystat-skills)"
    fi
}

echo "[1/3] Syncing shared references into shinystat and shinystat-cloud..."
if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] sync canonical src/medstat and shared references"
else
    if [[ -x "$REPO_ROOT/scripts/sync-skills.sh" || -f "$REPO_ROOT/scripts/sync-skills.sh" ]]; then
        bash "$REPO_ROOT/scripts/sync-skills.sh" >/dev/null
    else
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
        link_or_copy "$REPO_ROOT/src/medstat" "$REPO_ROOT/skills/shinystat/scripts/medstat"
    fi
fi

echo "[2/3] Installing cloud skill into packaging/cloud..."
if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$REPO_ROOT/packaging/cloud"
fi
link_or_copy "$REPO_ROOT/skills/shinystat-cloud" "$REPO_ROOT/packaging/cloud/shinystat-cloud"

echo "[3/3] Cleaning up legacy mirror directories..."

# Clean up legacy in-workspace mirror entries to maintain SSOT (ADR 49)
LEGACY_TARGETS=(
    "$REPO_ROOT/.agent/skills"
    "$REPO_ROOT/.agents/skills"
    "$REPO_ROOT/.claude/skills"
    "$REPO_ROOT/.cursor/skills"
    "$REPO_ROOT/.agent/agents"
    "$REPO_ROOT/.agents/agents"
)

for hdir in "${LEGACY_TARGETS[@]}"; do
    if [[ -L "$hdir" ]]; then
        remove_legacy_mirror "$hdir"
    elif [[ -d "$hdir" ]]; then
        for item in "$hdir"/*; do
            [[ -e "$item" || -L "$item" ]] || continue
            remove_legacy_mirror "$item"
        done
        if [[ "$DRY_RUN" != true ]]; then
            rmdir "$hdir" 2>/dev/null || true
            rmdir "$(dirname "$hdir")" 2>/dev/null || true
        fi
    fi
done

CANONICAL_AGENT="$REPO_ROOT/agents/figure-tester.md"

if [[ "$SCOPE" == "global" ]]; then
    echo "[Global] Installing standalone dereferenced skills into global config..."
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
        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] would run package-cloud-skill.sh"
        else
            "$REPO_ROOT/scripts/package-cloud-skill.sh" >/dev/null
        fi
    fi

    for gdir in "${GLOBAL_DIRS[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] would install into $gdir"
            continue
        fi
        mkdir -p "$gdir"
        rm -rf "$gdir/shinystat" "$gdir/shinystat-cloud"
        cp -RL "$REPO_ROOT/skills/shinystat" "$gdir/shinystat"
        cp -RL "$REPO_ROOT/skills/shinystat-cloud" "$gdir/shinystat-cloud"
        find "$gdir/shinystat" "$gdir/shinystat-cloud" -name ".DS_Store" -delete 2>/dev/null || true
        find "$gdir/shinystat" "$gdir/shinystat-cloud" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        echo "  ✓ Installed into $gdir"
    done

    # Global agent
    if [[ -f "$CANONICAL_AGENT" && -d "${HOME}/.gemini/config/agents" ]]; then
        if [[ "$TARGET" == "all" || "$TARGET" == "antigravity" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[dry-run] would install agent into ${HOME}/.gemini/config/agents/figure-tester.md"
            else
                cp "$CANONICAL_AGENT" "${HOME}/.gemini/config/agents/figure-tester.md"
                echo "  ✓ Installed figure-tester agent into ${HOME}/.gemini/config/agents/figure-tester.md"
            fi
        fi
    fi
fi

echo "Done! Installed shinystat-skills in '$MODE' mode (scope: $SCOPE)."
