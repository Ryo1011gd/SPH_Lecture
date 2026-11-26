#!/bin/bash
# 実験 (VULCANO VE-U7) と計算の先端位置・温度を比較する -> compare.png
cd "$(dirname "$0")"
python3 compare.py "$@"
