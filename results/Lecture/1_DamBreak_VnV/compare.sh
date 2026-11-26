#!/bin/bash
# 実験 (Martin & Moyce 1952) と計算の先端位置を比較する -> compare.png
cd "$(dirname "$0")"
python3 compare.py "$@"
