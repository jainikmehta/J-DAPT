#!/usr/bin/env bash
# ==============================================================================
# J-DAPT JailBreakV-28K Full Dataset Downloader & Extractor (~15.2 GB)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/datasets"
DOWNLOAD_DIR="${DATASET_DIR}/downloads"
OUTPUT_DIR="${DATASET_DIR}/jb28k"
ZIP_FILE="${DOWNLOAD_DIR}/JailBreakV_28K.zip"
LOG_FILE="${DOWNLOAD_DIR}/jb28k_download.log"
COOKIE_FILE="/tmp/gdrive_jb28k_cookie.txt"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${OUTPUT_DIR}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "${LOG_FILE}"
}

log "========================================================================"
log "🚀 Starting JailBreakV-28K Full Dataset Download (~15.2 GB)"
log "Working Directory : ${DATASET_DIR}"
log "Download Target   : ${ZIP_FILE}"
log "Extraction Target : ${OUTPUT_DIR}"
log "Log File          : ${LOG_FILE}"
log "========================================================================"

URL_INITIAL="https://drive.usercontent.google.com/download?id=1ZrvSHklXiGYhpiVoxUH8FWc5k0fv2xVZ&export=download&authuser=0"
URL_CONFIRM="https://drive.usercontent.google.com/download?id=1ZrvSHklXiGYhpiVoxUH8FWc5k0fv2xVZ&export=download&authuser=0&confirm=t"

log "📥 Initializing Google Drive session cookies..."
curl -c "${COOKIE_FILE}" -s -L "${URL_INITIAL}" > /dev/null

log "📥 Starting download of JailBreakV_28K.zip..."
curl -b "${COOKIE_FILE}" -C - -L \
    -# \
    -o "${ZIP_FILE}" \
    "${URL_CONFIRM}" 2>&1 | tr '\r' '\n' | tee -a "${LOG_FILE}"

log "✅ Download complete! File size: $(ls -lh "${ZIP_FILE}" | awk '{print $5}')"

log "📦 Extracting JailBreakV_28K.zip into ${OUTPUT_DIR}..."
unzip -q -o "${ZIP_FILE}" -d "${OUTPUT_DIR}" || true

log "========================================================================"
log "🎉 JailBreakV-28K Download & Extraction Finished Successfully!"
log "========================================================================"

