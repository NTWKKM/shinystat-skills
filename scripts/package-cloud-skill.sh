#!/usr/bin/env bash
# scripts/package-cloud-skill.sh
# Builds standalone shinystat-cloud artifact with dereferenced symlinks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_SKILL="$REPO_ROOT/skills/shinystat-cloud"
DIST_DIR="$REPO_ROOT/packaging/cloud"
TARGET_SKILL="$DIST_DIR/shinystat-cloud"

echo "[1/3] Preparing clean build directory at $TARGET_SKILL..."
rm -rf "$TARGET_SKILL"
mkdir -p "$DIST_DIR"

# Ensure local symlinks (shared references) are up to date before dereferencing
"$REPO_ROOT/scripts/install-skills.sh" --symlink >/dev/null

# Copy with -L to dereference all symlinks into standalone physical files
cp -RL "$SRC_SKILL" "$TARGET_SKILL"

echo "[2/3] Generating combined python-recipes.md for snapshot/backward compatibility..."
RECIPES_DIR="$TARGET_SKILL/references/recipes"
COMBINED_RECIPES="$TARGET_SKILL/references/python-recipes.md"

if [[ -d "$RECIPES_DIR" ]]; then
    {
        echo "# ShinyStat Cloud — Combined Python Recipes (Auto-generated)"
        echo ""
        echo "> **Note for Agents:** Prefer reading modular files in \`references/recipes/\` to save context tokens."
        echo ""
        for part in "$RECIPES_DIR"/*.md; do
            [[ -e "$part" ]] || continue
            cat "$part"
            echo -e "\n\n---\n"
        done
    } > "$COMBINED_RECIPES"
    # Sync back to source reference if unit tests inspect skills/shinystat-cloud directly
    cp "$COMBINED_RECIPES" "$SRC_SKILL/references/python-recipes.md"
fi

echo "[3/3] Creating distributable zip archive..."
(cd "$DIST_DIR" && zip -rq "shinystat-cloud.zip" "shinystat-cloud")

echo "Cloud skill packaged successfully at: $DIST_DIR/shinystat-cloud.zip"
