#!/bin/bash
# 粒子を作り (box.grid)、コンクリート粒子に剛体ラベルを付ける (box.rigid)
cd "$(dirname "$0")"
../../../generator/GeneratorForMph box
python3 make_rigid.py
