#!/bin/bash
# 引数: data grid prof vtk log スレッド数 (12 コアのマシン向け。コア数に合わせて変える)
# (ソルバは source で make CPU=1 を実行して作る -> source/SPH_omp)
cd "$(dirname "$0")"
mkdir -p output
../../../source/SPH_omp box.data box.grid output/box%04d.prof output/box%04d.vtk output/box.log 12
