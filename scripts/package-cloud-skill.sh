#!/usr/bin/env bash
# ==============================================================================
# package-cloud-skill.sh — Package dedicated Cloud Skill for Claude Web
# Builds a self-contained shinystat-cloud.zip package for claude.ai
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOUD_SRC="${REPO_ROOT}/packaging/cloud"
DEST_DIR="${1:-${HOME}/Desktop}"

if [[ ! -d "${CLOUD_SRC}/shinystat-cloud" ]]; then
    echo "Error: Cloud skill source not found at '${CLOUD_SRC}/shinystat-cloud'" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
OUTPUT_ZIP="${DEST_DIR}/shinystat-cloud.zip"

echo "=========================================================="
echo "  Packaging Shinystat Cloud Skill for Claude Web"
echo "=========================================================="
echo "Source: ${CLOUD_SRC}/shinystat-cloud"
echo "Destination: ${OUTPUT_ZIP}"

(cd "${CLOUD_SRC}" && zip -r "${OUTPUT_ZIP}" shinystat-cloud -x "*.DS_Store" -x "*__pycache__*")

echo ""
echo "✓ Successfully created: ${OUTPUT_ZIP}"
echo "=========================================================="
