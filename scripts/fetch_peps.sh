#!/usr/bin/env bash
# Fetch the subset of Python PEPs used as this project's corpus.
# Run once: bash scripts/fetch_peps.sh  (or: make fetch-peps)
set -e
DEST="data/corpus"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
echo "Cloning python/peps..."
git clone --depth 1 --single-branch https://github.com/python/peps.git "$TMP" >/dev/null 2>&1
PEPS="0008 0020 0257 0263 0289 0318 0328 0343 0380 0405 0440 0484 0492 0498 0518 0526 0557 0572 0585 0604 0612 0621 0634 0636 0695 0701 0703"
for n in $PEPS; do
  cp "$TMP/peps/pep-$n.rst" "$DEST/" 2>/dev/null || echo "  skip pep-$n (not found)"
done
rm -rf "$TMP"
echo "Copied $(ls "$DEST"/pep-*.rst 2>/dev/null | wc -l | tr -d ' ') PEPs into $DEST/"
