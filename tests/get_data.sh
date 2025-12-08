#!/usr/bin/env bash
# Wrapper script for cross-platform compatibility
# Calls the Python version which works on Linux, Mac, and Windows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/get_data.py"
