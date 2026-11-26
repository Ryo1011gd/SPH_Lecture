#!/bin/bash
# Export PPE training snapshots from existing case directories (normal Jacobi runs).
# Usage: ./gen_snapshots.sh SOLVER OUTDIR ENDTIME EVERY CASE_DIR [CASE_DIR ...]
#   e.g. ./gen_snapshots.sh ../SPH_acc data 0.5 50 ../../results/GNS_HighViscosity_VnV/cases/D01_*
# Each case is run in OUTDIR/<case name>/ with its box.data/box.grid; snapshots go to snaps/.
set -u
BIN=$(readlink -f "$1"); OUT=$(readlink -f "$2"); END=$3; EVERY=$4; shift 4
for c in "$@"; do
    c=$(readlink -f "$c"); name=$(basename "$c"); d="$OUT/$name"
    mkdir -p "$d" && cd "$d" || exit 1
    ln -sf "$c/box.grid" .
    sed -e "s/^EndTime.*/EndTime $END/" -e 's/^OutputInterval.*/OutputInterval 100/' \
        -e 's/^VtkOutputInterval.*/VtkOutputInterval 100/' "$c/box.data" | grep -v '^PpeDon' > box.data
    printf "PpeDonExportEvery %s\nPpeDonExportDir snaps\n" "$EVERY" >> box.data
    s=$(date +%s)
    "$BIN" box.data box.grid box%03d.prof box%03d.vtk box.log 4 > out.txt 2>&1
    echo "$name rc=$? $(( $(date +%s)-s ))s $(ls snaps 2>/dev/null | wc -l) snapshots"
done
