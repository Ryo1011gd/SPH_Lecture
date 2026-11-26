#!/bin/bash
# 理論解 (静水圧) と計算結果を比較する -> compare.png
cd "$(dirname "$0")"
python3 compare.py "$@"
