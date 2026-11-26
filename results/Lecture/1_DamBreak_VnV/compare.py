#!/usr/bin/env python3
"""演習1: ダムブレイクの先端位置を Martin & Moyce (1952) の実験と比較する。

使い方:
    python3 compare.py                 # output/ の結果を比較
    python3 compare.py output run_dp2.5mm/output   # 複数の結果を重ねる (解像度の比較など)
出力: compare.png (図), front.csv (先端位置の時刻歴)
必要なもの: Python3, numpy, matplotlib
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = 0.1      # 水柱の初期幅 a [m] (box.boid と同じ)
G = 9.81

# Martin & Moyce (1952), n^2 = 2 の先端位置 (粒子法の論文で広く使われている値)
#   T = t sqrt(2g/a),  Z = z/a
EXP_T = [0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.20, 2.32, 2.51, 2.65, 2.83, 2.98, 3.11, 3.33]
EXP_Z = [1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67, 3.89, 4.11]


def read_front(outdir):
    """prof ファイル (1 行 = type x y z u v w ...) から、流体 (type 0,1) の最も右の x を取り出す。"""
    t, xf = [], []
    for f in sorted(glob.glob(os.path.join(outdir, "*.prof"))):
        with open(f) as fp:
            time = float(fp.readline())
            fp.readline()
            a = np.loadtxt(fp, ndmin=2)
        fl = a[:, 0] <= 1
        t.append(time)
        xf.append(a[fl, 1].max())
    o = np.argsort(t)                 # ファイル名はステップ番号なので時刻で並べ直す
    return np.array(t)[o], np.array(xf)[o]


def main():
    dirs = sys.argv[1:] or ["output"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(EXP_T, EXP_Z, "ks", mfc="none", label="Martin & Moyce (1952)")
    print(f"{'結果':30s} {'RMS誤差 (Z)':>12s}")
    for d in dirs:
        t, xf = read_front(d)
        T = t * np.sqrt(2 * G / A)
        Z = xf / A
        ax.plot(T, Z, "-", label=f"SPH ({d})")
        sel = np.array(EXP_T) <= T.max()
        rms = np.sqrt(np.mean((np.interp(np.array(EXP_T)[sel], T, Z) - np.array(EXP_Z)[sel]) ** 2))
        print(f"{d:30s} {rms:12.3f}")
        np.savetxt("front.csv" if len(dirs) == 1 else f"front_{d.replace('/', '_')}.csv",
                   np.column_stack([t, xf, T, Z]), delimiter=",",
                   header="t[s],x_front[m],T,Z", comments="")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(1, 5)
    ax.set_xlabel(r"$T = t\sqrt{2g/a}$")
    ax.set_ylabel(r"$Z = z/a$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("compare.png", dpi=150)
    print("-> compare.png")


if __name__ == "__main__":
    main()
