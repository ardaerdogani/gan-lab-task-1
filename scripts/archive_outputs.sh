#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(date +"%Y%m%d-%H%M%S")}"
ARCHIVE_DIR="$ROOT_DIR/archives/$STAMP"

mkdir -p "$ARCHIVE_DIR"

move_if_exists() {
  local path="$1"
  if [ -e "$ROOT_DIR/$path" ]; then
    mv "$ROOT_DIR/$path" "$ARCHIVE_DIR/"
    printf 'Archived %s -> %s\n' "$path" "archives/$STAMP/"
  fi
}

move_if_exists "runs"
move_if_exists "data_synth"
move_if_exists "reports"

printf 'Archive ready: %s\n' "$ARCHIVE_DIR"
