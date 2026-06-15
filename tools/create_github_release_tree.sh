#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/release_artifact
cp README.md LICENSE CITATION.cff outputs/release_artifact/
cp -r configs docs scripts src tests outputs/release_artifact/
find outputs/release_artifact -type d -name __pycache__ -prune -exec rm -rf {} +
echo "release artifact staged in outputs/release_artifact"
