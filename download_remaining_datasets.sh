#!/usr/bin/env bash
# ==============================================================================
# J-DAPT Dataset Downloader & Extractor
# Downloads:
#   1. ABOships-PLUS (~8.1 GB)
#   2. LaRS (~14.6 GB)
#   3. nuScenes v1.0-trainval01 blobs (~31.6 GB)
#   4. nuScenes v1.0-trainval metadata (~461 MB)
# Also unzips Wayve images (~8.3 GB)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/datasets"
DOWNLOAD_DIR="${DATASET_DIR}/downloads"
LOG_FILE="${DOWNLOAD_DIR}/download.log"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${DATASET_DIR}/aboships"
mkdir -p "${DATASET_DIR}/lars"
mkdir -p "${DATASET_DIR}/nuscenes"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "${LOG_FILE}"
}

log "========================================================================"
log "🚀 Starting J-DAPT Remaining Datasets Download"
log "Working Directory : ${DATASET_DIR}"
log "Download Cache    : ${DOWNLOAD_DIR}"
log "Log File          : ${LOG_FILE}"
log "========================================================================"

# Function to download with resume
download_file() {
    local url="$1"
    local output_path="$2"
    local desc="$3"

    log "------------------------------------------------------------------------"
    log "📥 Downloading ${desc}..."
    log "URL: ${url}"
    log "Target: ${output_path}"

    wget -c \
        --progress=dot:giga \
        --tries=20 \
        --waitretry=5 \
        --timeout=30 \
        -O "${output_path}" \
        "${url}" 2>&1 | tee -a "${LOG_FILE}"

    log "✅ Finished downloading ${desc}."
}

# ------------------------------------------------------------------------------
# 1. ABOships-PLUS (~8.1 GB)
# ------------------------------------------------------------------------------
ABOSHIPS_URL="https://zenodo.org/records/8383205/files/ABOships-PLUS.zip?download=1"
ABOSHIPS_ZIP="${DOWNLOAD_DIR}/ABOships-PLUS.zip"
download_file "${ABOSHIPS_URL}" "${ABOSHIPS_ZIP}" "ABOships-PLUS"

log "📦 Extracting ABOships-PLUS..."
unzip -q -o "${ABOSHIPS_ZIP}" -d "${DATASET_DIR}/aboships" || true
log "✅ ABOships-PLUS extracted."

# ------------------------------------------------------------------------------
# 2. LaRS (~14.6 GB)
# ------------------------------------------------------------------------------
LARS_URL="https://box.vicos.si/lars/lars_v1.0.0_images_seq.zip"
LARS_ZIP="${DOWNLOAD_DIR}/lars_v1.0.0_images_seq.zip"
download_file "${LARS_URL}" "${LARS_ZIP}" "LaRS"

log "📦 Extracting LaRS..."
unzip -q -o "${LARS_ZIP}" -d "${DATASET_DIR}/lars" || true
log "✅ LaRS extracted."

# ------------------------------------------------------------------------------
# 3. nuScenes Metadata (~461 MB)
# ------------------------------------------------------------------------------
NUSC_META_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval_meta.tgz"
NUSC_META_TGZ="${DOWNLOAD_DIR}/v1.0-trainval_meta.tgz"
download_file "${NUSC_META_URL}" "${NUSC_META_TGZ}" "nuScenes Metadata"

log "📦 Extracting nuScenes Metadata..."
tar -xzf "${NUSC_META_TGZ}" -C "${DATASET_DIR}/nuscenes" || true
log "✅ nuScenes Metadata extracted."

# ------------------------------------------------------------------------------
# 4. nuScenes TrainVal Blobs 01 (~31.6 GB)
# ------------------------------------------------------------------------------
NUSC_BLOBS_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
NUSC_BLOBS_TGZ="${DOWNLOAD_DIR}/v1.0-trainval01_blobs.tgz"
download_file "${NUSC_BLOBS_URL}" "${NUSC_BLOBS_TGZ}" "nuScenes TrainVal Blobs 01"

log "📦 Extracting nuScenes TrainVal Blobs 01..."
tar -xzf "${NUSC_BLOBS_TGZ}" -C "${DATASET_DIR}/nuscenes" || true
log "✅ nuScenes TrainVal Blobs 01 extracted."

# ------------------------------------------------------------------------------
# 5. Extract Wayve images if not yet extracted
# ------------------------------------------------------------------------------
if [ -f "${DATASET_DIR}/Wayve/images.zip" ] && [ ! -d "${DATASET_DIR}/Wayve/images" ]; then
    log "📦 Extracting Wayve images.zip..."
    unzip -q -o "${DATASET_DIR}/Wayve/images.zip" -d "${DATASET_DIR}/Wayve" || true
    log "✅ Wayve images extracted."
fi

log "========================================================================"
log "🎉 ALL DATASETS DOWNLOADED AND EXTRACTED SUCCESSFULLY!"
log "========================================================================"

