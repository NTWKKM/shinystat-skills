#!/usr/bin/env bash
# scripts/package-cloud-skill.sh
# Builds standalone shinystat-cloud artifact with dereferenced symlinks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_SKILL="$REPO_ROOT/skills/shinystat-cloud"
DIST_DIR="$REPO_ROOT/packaging/cloud"
TARGET_SKILL="$DIST_DIR/shinystat-cloud"

echo "[1/3] Preparing clean build directory at $TARGET_SKILL..."
mkdir -p "$DIST_DIR"

# Ensure local symlinks (shared references) are up to date before dereferencing
"$REPO_ROOT/scripts/install-skills.sh" --symlink >/dev/null

# Remove TARGET_SKILL (which install-skills.sh created as a symlink) before copying so cp -RL creates a standalone directory
rm -rf "$TARGET_SKILL"

# Copy with -L to dereference all symlinks into standalone physical files
cp -RL "$SRC_SKILL" "$TARGET_SKILL"
rm -f "$TARGET_SKILL/references/python-recipes.md" "$SRC_SKILL/references/python-recipes.md"

echo "[2/3] Rewriting shared-reference links in packaged Markdown files..."
TARGET_SKILL="$TARGET_SKILL" python3 - <<'PY'
import os
from pathlib import Path

target_dir = Path(os.environ["TARGET_SKILL"])
for md_file in target_dir.rglob("*.md"):
    if not md_file.is_file():
        continue
    content = md_file.read_text(encoding="utf-8")
    ref_rel = os.path.relpath(target_dir / "references", md_file.parent)
    prefix = "" if ref_rel == "." else f"{ref_rel}/"

    new_content = content
    new_content = new_content.replace("../../../_shared/references/", prefix)
    new_content = new_content.replace("../../_shared/references/", prefix)
    new_content = new_content.replace("../_shared/references/", prefix)
    new_content = new_content.replace("(`../_shared/references/`)", f"(`{prefix}`)")

    if new_content != content:
        md_file.write_text(new_content, encoding="utf-8")
PY

echo "[3/3] Creating distributable zip archive..."
rm -f "$DIST_DIR/shinystat-cloud.zip"
(cd "$DIST_DIR" && zip -rq "shinystat-cloud.zip" "shinystat-cloud" -x "*.DS_Store" -x "*__pycache__*")

echo "Cloud skill packaged successfully at: $DIST_DIR/shinystat-cloud.zip"
