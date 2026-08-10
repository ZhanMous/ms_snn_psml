#!/usr/bin/env bash
set -euo pipefail

PSML_URL="https://zenodo.org/record/5130612/files/PSML.zip?download=1"
DEST_DIR="${1:-data}"
ARCHIVE_NAME="${2:-PSML.zip}"
ARCHIVE_PATH="${DEST_DIR}/${ARCHIVE_NAME}"

mkdir -p "${DEST_DIR}"

echo "Downloading PSML from official Zenodo URL:"
echo "  ${PSML_URL}"
echo "Archive:"
echo "  ${ARCHIVE_PATH}"

if command -v curl >/dev/null 2>&1; then
  curl -L -o "${ARCHIVE_PATH}" "${PSML_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE_PATH}" "${PSML_URL}"
else
  echo "Neither curl nor wget is available." >&2
  exit 1
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -n "${ARCHIVE_PATH}" -d "${DEST_DIR}"
elif command -v 7z >/dev/null 2>&1; then
  7z x -y "${ARCHIVE_PATH}" "-o${DEST_DIR}"
else
  echo "Downloaded archive, but neither unzip nor 7z is available." >&2
  exit 1
fi

echo "PSML archive extracted under:"
echo "  ${DEST_DIR}"

if [ -d "${DEST_DIR}/PSML" ]; then
  echo "PSML extracted. The processed classification split used by the"
  echo "experiments must be available at:"
  echo "  ${DEST_DIR}/PSML/processed_dataset/classification.pkl"
else
  echo "Inspect the extracted dataset root with:"
  echo "  ls ${DEST_DIR}"
fi
