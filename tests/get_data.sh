#!/usr/bin/env bash
set -euo pipefail

ISMIP_HOM_OGA_URL="https://frank.pattyn.web.ulb.be/ismip/tc-2-95-2008-supplement.zip"
ISMIP_HOM_AROLLA_URL="https://frank.pattyn.web.ulb.be/ismip/arolla100.dat"
ISMIP_HOM_OGA_ZIP_INNER="tc-2007-0019-sp2.zip"
ISMIP_HOM_OGA_ZIP_OUTER="tc-2-95-2008-supplement.zip"
ISMIP_HOM_TARGET_DIR="./test_iceflow/ismip_hom/data"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "Using temp dir: $tmpdir"

# Download ISMIP-HOM OGA reference data
wget -O "$tmpdir/$ISMIP_HOM_OGA_ZIP_OUTER" "$ISMIP_HOM_OGA_URL"

unzip -q "$tmpdir/$ISMIP_HOM_OGA_ZIP_OUTER" -d "$tmpdir"
unzip -q "$tmpdir/$ISMIP_HOM_OGA_ZIP_INNER" -d "$tmpdir"

mkdir -p "$ISMIP_HOM_TARGET_DIR"
rm -rf "$ISMIP_HOM_TARGET_DIR/oga"

cp -a "$tmpdir/ismip_all/oga" "$ISMIP_HOM_TARGET_DIR/"

echo "✅ ISMIP-HOM OGA reference data downloaded to <$ISMIP_HOM_TARGET_DIR/oga>"

# Download ISMIP-HOM Arolla input data
mkdir -p "$ISMIP_HOM_TARGET_DIR/arolla"
wget -O "$ISMIP_HOM_TARGET_DIR/arolla/arolla100.dat" "$ISMIP_HOM_AROLLA_URL"

echo "✅ ISMIP-HOM Arolla input data downloaded to <$ISMIP_HOM_TARGET_DIR/arolla>"
