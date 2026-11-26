#!/usr/bin/env python3
"""演習3: VULCANO VE-U7 (セラミック流路) のコリウム先端位置を実験と比較する。

使い方:
    python3 compare.py                         # output/ の結果
    python3 compare.py ref/output visc3/output k80/output rad5/output   # 感度解析を重ねる
出力: compare.png (先端位置と平均温度), front.csv
必要なもの: Python3, numpy, matplotlib
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# VE-U7 セラミック流路の先端位置。Jubaidah 博士論文 (2021) Fig. 21 の実験点を図から読み取った値
# (読み取り誤差 ±0.01 m, ±0.3 s 程度)。元データは Journeau et al. (2006)。
EXP_T = [0.2, 0.7, 1.7, 2.2, 3.8, 4.2, 5.6, 7.4, 9.2, 21.6, 24.0]
EXP_X = [0.071, 0.121, 0.172, 0.192, 0.223, 0.272, 0.323, 0.371, 0.424, 0.456, 0.456]
FINAL_CERAMIC = 0.45     # 最終拡がり長さ [m] (セラミック)
FINAL_CONCRETE = 0.36    # 最終拡がり長さ [m] (コンクリート、参考)
POUR_END = 8.0


def read(outdir):
    t, xf, Tm = [], [], []
    for f in sorted(glob.glob(os.path.join(outdir, "*.prof"))):
        with open(f) as fp:
            time = float(fp.readline())
            fp.readline()
            a = np.loadtxt(fp, ndmin=2)
        melt = a[:, 0] == 1
        t.append(time)
        xf.append(max(0.0, a[melt, 1].max()))
        Tm.append(a[melt, 7].mean() if a.shape[1] > 7 else np.nan)
    o = np.argsort(t)                 # ファイル名はステップ番号なので時刻で並べ直す
    return np.array(t)[o], np.array(xf)[o], np.array(Tm)[o]


def main():
    dirs = sys.argv[1:] or ["output"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax.plot(EXP_T, EXP_X, "k^", label="VE-U7 exp. (ceramic)")
    ax.axhline(FINAL_CERAMIC, color="k", ls=":", lw=1)
    ax.axhline(FINAL_CONCRETE, color="gray", ls=":", lw=1)
    ax.text(14, FINAL_CONCRETE + 0.01, "final, concrete channel", color="gray")
    ax.axvline(POUR_END, color="gray", ls="--", lw=1)
    print(f"{'結果':30s} {'最終位置[m]':>10s} {'実験との差[m]':>12s} {'実験点との RMS[m]':>14s}")
    for d in dirs:
        t, xf, Tm = read(d)
        ax.plot(t, xf, "-", label=d)
        ax2.plot(t, Tm, "-", label=d)
        # 実験点 (計算した時間の範囲内) での先端位置の差の RMS
        err = [np.interp(te, t, xf) - xe for te, xe in zip(EXP_T, EXP_X) if te <= t[-1]]
        rms = np.sqrt(np.mean(np.square(err)))
        print(f"{d:30s} {xf[-1]:10.3f} {xf[-1] - FINAL_CERAMIC:12.3f} {rms:14.3f}   (t = {t[-1]:.0f} s)")
        np.savetxt("front.csv" if len(dirs) == 1 else f"front_{os.path.basename(os.path.dirname(os.path.abspath(d)))}.csv",
                   np.column_stack([t, xf, Tm]), delimiter=",",
                   header="t[s],x_front[m],T_mean_melt[K]", comments="")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("spreading edge [m]")
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax2.axhline(2623, color="r", ls=":", lw=1)
    ax2.text(1, 2600, "liquidus 2623 K", color="r")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("mean corium temperature [K]")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("compare.png", dpi=150)
    print("-> compare.png")


if __name__ == "__main__":
    main()
