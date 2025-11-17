#!/usr/bin/env bash
set -euo pipefail

URL="https://frank.pattyn.web.ulb.be/ismip/tc-2-95-2008-supplement.zip"
ZIP_INNER="tc-2007-0019-sp2.zip"
ZIP_OUTER="tc-2-95-2008-supplement.zip"
TARGET_DIR="./test_iceflow/ismip_hom/data"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "Using temp dir: $tmpdir"

wget -O "$tmpdir/$ZIP_OUTER" "$URL"

unzip -q "$tmpdir/$ZIP_OUTER" -d "$tmpdir"
unzip -q "$tmpdir/$ZIP_INNER" -d "$tmpdir"

mkdir -p "$TARGET_DIR"
rm -rf "$TARGET_DIR/oga"

cp -a "$tmpdir/ismip_all/oga" "$TARGET_DIR/"

echo "✅ Data successfully downloaded and put in <$TARGET_DIR/oga>."
