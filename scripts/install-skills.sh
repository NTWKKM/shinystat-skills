#!/usr/bin/env bash
# scripts/install-skills.sh
# Single-Source-of-Truth installer for shinystat-skills (Phase 1 & Phase 3)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---symlink}" # รองรับ --symlink (ค่าเริ่มต้น) หรือ --copy

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

echo "[1/4] Syncing shared references into shinystat and shinystat-cloud..."
SHARED_REF_DIR="$REPO_ROOT/skills/_shared/references"
if [[ -d "$SHARED_REF_DIR" ]]; then
    for ref_file in "$SHARED_REF_DIR"/*.md; do
        [[ -e "$ref_file" ]] || continue
        fname="$(basename "$ref_file")"
        link_or_copy "$ref_file" "$REPO_ROOT/skills/shinystat/references/$fname"
        link_or_copy "$ref_file" "$REPO_ROOT/skills/shinystat-cloud/references/$fname"
    done
fi

echo "[2/4] Linking canonical src/medstat into skills/shinystat/scripts/medstat..."
link_or_copy "$REPO_ROOT/src/medstat" "$REPO_ROOT/skills/shinystat/scripts/medstat"

echo "[3/4] Installing skills into IDE harness directories (.agent, .agents, .claude, .cursor)..."
HARNESS_DIRS=(
    "$REPO_ROOT/.agent/skills"
    "$REPO_ROOT/.agents/skills"
    "$REPO_ROOT/.claude/skills"
    "$REPO_ROOT/.cursor/skills"
)

for hdir in "${HARNESS_DIRS[@]}"; do
    mkdir -p "$hdir"
    link_or_copy "$REPO_ROOT/skills/shinystat" "$hdir/shinystat"
    link_or_copy "$REPO_ROOT/skills/shinystat-cloud" "$hdir/shinystat-cloud"
    if [[ "$MODE" == "--copy" ]]; then
        link_or_copy "$REPO_ROOT/skills/_shared" "$hdir/_shared"
    else
        rm -rf "$hdir/_shared"
    fi
done

echo "[3.5/4] Installing cloud skill into packaging/cloud..."
mkdir -p "$REPO_ROOT/packaging/cloud"
link_or_copy "$REPO_ROOT/skills/shinystat-cloud" "$REPO_ROOT/packaging/cloud/shinystat-cloud"
if [[ "$MODE" == "--copy" ]]; then
    link_or_copy "$REPO_ROOT/skills/_shared" "$REPO_ROOT/packaging/cloud/_shared"
else
    rm -rf "$REPO_ROOT/packaging/cloud/_shared"
fi

echo "[4/4] Linking sub-agents (figure-tester.md)..."
CANONICAL_AGENT="$REPO_ROOT/agents/figure-tester.md"
if [[ -f "$CANONICAL_AGENT" ]]; then
    link_or_copy "$CANONICAL_AGENT" "$REPO_ROOT/.agent/agents/figure-tester.md"
    link_or_copy "$CANONICAL_AGENT" "$REPO_ROOT/.agents/agents/figure-tester.md"
fi

echo "Done! Installed shinystat-skills in '$MODE' mode."
