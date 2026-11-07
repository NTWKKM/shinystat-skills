#!/usr/bin/env bash
# scripts/sync-skills.sh
# Synchronizes canonical SSOT (src/medstat and skills/_shared/references)
# into self-contained physical directories:
#   - skills/shinystat/references/
#   - skills/shinystat/scripts/medstat/
#   - skills/shinystat-cloud/references/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --check)
            CHECK_ONLY=true
            ;;
        -h|--help)
            echo "Usage: $0 [--check]"
            echo "  --check   Validate skills are synchronized with SSOT without uncommitted diffs"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

echo "[1/3] Syncing shared references into skills/shinystat/references/..."
SHARED_REF_DIR="$REPO_ROOT/skills/_shared/references"
SHINYSTAT_REF_DIR="$REPO_ROOT/skills/shinystat/references"
mkdir -p "$SHINYSTAT_REF_DIR"

for ref_file in "$SHARED_REF_DIR"/*.md; do
    [[ -e "$ref_file" ]] || continue
    fname="$(basename "$ref_file")"
    rm -rf "$SHINYSTAT_REF_DIR/$fname"
    cp "$ref_file" "$SHINYSTAT_REF_DIR/$fname"
done

echo "[2/3] Syncing canonical src/medstat into skills/shinystat/scripts/medstat/..."
SHINYSTAT_SCRIPTS_DIR="$REPO_ROOT/skills/shinystat/scripts"
TARGET_MEDSTAT="$SHINYSTAT_SCRIPTS_DIR/medstat"
mkdir -p "$SHINYSTAT_SCRIPTS_DIR"
rm -rf "$TARGET_MEDSTAT"
cp -R "$REPO_ROOT/src/medstat" "$TARGET_MEDSTAT"

# Remove __pycache__ and .DS_Store from vendored copy
find "$TARGET_MEDSTAT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TARGET_MEDSTAT" -type f -name ".DS_Store" -delete 2>/dev/null || true

echo "[3/3] Syncing shared references into skills/shinystat-cloud/references/..."
CLOUD_REF_DIR="$REPO_ROOT/skills/shinystat-cloud/references"
mkdir -p "$CLOUD_REF_DIR"

for ref_file in "$SHARED_REF_DIR"/*.md; do
    [[ -e "$ref_file" ]] || continue
    fname="$(basename "$ref_file")"
    # Never overwrite cloud-native standalone figures.md
    if [[ "$fname" == "figures.md" ]]; then
        continue
    fi
    rm -rf "$CLOUD_REF_DIR/$fname"
    cp "$ref_file" "$CLOUD_REF_DIR/$fname"
done

# General cleanup in skills/
find "$REPO_ROOT/skills" -type f -name ".DS_Store" -delete 2>/dev/null || true
find "$REPO_ROOT/skills" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "✓ Skills synchronized successfully."

if [[ "$CHECK_ONLY" == true ]]; then
    echo "Verifying zero git drift in skills/..."
    if ! git -C "$REPO_ROOT" diff --exit-code skills/; then
        echo "Error: skills/ is out of sync with SSOT (src/medstat or skills/_shared/references)." >&2
        echo "Run './scripts/sync-skills.sh' and commit the changes." >&2
        exit 1
    fi
    echo "✓ Zero git drift verified in skills/."
fi
