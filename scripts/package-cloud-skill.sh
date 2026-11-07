#!/usr/bin/env bash
# scripts/package-cloud-skill.sh
# Packages standalone shinystat-cloud skill into distributable directory and zip archive
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_SKILL="$REPO_ROOT/skills/shinystat-cloud"
DIST_DIR="$REPO_ROOT/packaging/cloud"
TARGET_SKILL="$DIST_DIR/shinystat-cloud"

echo "[1/3] Ensuring canonical skills are synchronized..."
"$REPO_ROOT/scripts/sync-skills.sh" >/dev/null

echo "[2/3] Preparing clean build directory at $TARGET_SKILL..."
mkdir -p "$DIST_DIR"
rm -rf "$TARGET_SKILL"
cp -R "$SRC_SKILL" "$TARGET_SKILL"

# Clean any OS or Python artifacts
find "$TARGET_SKILL" -name ".DS_Store" -delete 2>/dev/null || true
find "$TARGET_SKILL" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "[3/3] Creating distributable zip archive..."
rm -f "$DIST_DIR/shinystat-cloud.zip"
(cd "$DIST_DIR" && zip -rq "shinystat-cloud.zip" "shinystat-cloud" -x "*.DS_Store" -x "*__pycache__*")

echo "Cloud skill packaged successfully at: $DIST_DIR/shinystat-cloud.zip"
