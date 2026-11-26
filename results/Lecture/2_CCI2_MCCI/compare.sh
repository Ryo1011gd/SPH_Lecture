#!/bin/bash
# CCI-2 実験 (侵食深さ) と計算を比較する -> compare.png, snap.png
cd "$(dirname "$0")"
python3 compare.py "$@"
