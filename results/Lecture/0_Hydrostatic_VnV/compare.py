#!/usr/bin/env python3
"""演習0: 静水圧 (Verification) — 計算した圧力を理論解 p = rho g (H - y) と比較する。

使い方: ./compare.sh   (= python3 compare.py)
出力: compare.png
  左: 最後の時刻の圧力 vs 高さ (点 = 粒子、線 = 理論解)
  右: 底の圧力 (理論 rho g H との比) と 最大速度 の時刻歴 (静止のままか)
必要なもの: Python3, numpy, matplotlib
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RHO = 1000.0   # box.data の Density (Type 1)
G = 9.81
H = 0.20       # 水深 [m] (box.boid)
DT_VTK = 0.05  # box.data の VtkOutputInterval


def read_vtk(f):
    """VTK から 位置, 種類 (label), 圧力 (PressureP) を読む。"""
    with open(f) as fp:
        lines = fp.read().split("\n")
    n = int(lines[4].split()[1])
    pos = np.array([l.split()[:3] for l in lines[5:5 + n]], dtype=float)
    out = {}
    i = 5 + n
    while i < len(lines):
        if lines[i].startswith("SCALARS"):
            out[lines[i].split()[1]] = np.array(lines[i + 2:i + 2 + n], dtype=float)
            i += 2 + n
        else:
            i += 1
    return pos, out["label"], out["PressureP"]


def read_prof(f):
    with open(f) as fp:
        t = float(fp.readline())
        fp.readline()
        d = np.loadtxt(fp, ndmin=2)
    return t, d


def main():
    vtks = sorted(glob.glob(os.path.join("output", "*.vtk")))
    profs = sorted(glob.glob(os.path.join("output", "*.prof")))
    if not vtks:
        raise SystemExit("output/*.vtk がない。先に ./execute.sh を実行する")
    p_theory_bottom = RHO * G * H

    # 最後の時刻の圧力分布
    pos, lab, p = read_vtk(vtks[-1])
    w = lab == 1
    y, pw = pos[w, 1], p[w]
    p_th = RHO * G * (H - y)
    err = pw - p_th
    rms = np.sqrt(np.mean(err ** 2)) / p_theory_bottom * 100
    inner = (pos[w, 0] > 0.05) & (pos[w, 0] < 0.35)             # 壁から離れた所
    rms_in = np.sqrt(np.mean(err[inner] ** 2)) / p_theory_bottom * 100
    # 傾き dp/dy (理論 -rho g) と、表面の実際の位置 Hs (一番上の粒子の中心, p = 0 を課す所) から測った誤差
    slope, icpt = np.polyfit(y[inner], pw[inner], 1)
    Hs = np.mean(np.sort(y)[-int(0.40 / 0.005):])      # 一番上の 1 列 (幅 0.4 m / dp) の平均高さ
    rms_s = np.sqrt(np.mean((pw - RHO * G * (Hs - y)) ** 2)) / p_theory_bottom * 100

    # 時刻歴: 底付近 (y < 2 dp) の平均圧力, 最大速度
    tt, pb, vmax = [], [], []
    for f in vtks:
        _, lb, pp = read_vtk(f)
        pos_f = read_vtk(f)[0]
        m = (lb == 1) & (pos_f[:, 1] < 0.01) & (pos_f[:, 0] > 0.05) & (pos_f[:, 0] < 0.35)
        yb = pos_f[m, 1].mean()
        tt.append(len(tt) * DT_VTK)
        pb.append(pp[m].mean() / (RHO * G * (H - yb)))
    tv = []
    for f in profs:
        t, d = read_prof(f)
        m = d[:, 0] == 1
        tv.append(t)
        vmax.append(np.sqrt(d[m, 4] ** 2 + d[m, 5] ** 2).max())
    o = np.argsort(tv)
    tv, vmax = np.array(tv)[o], np.array(vmax)[o]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax.plot(pw, y, ".", ms=2, alpha=0.4, label="SPH (particles)")
    yy = np.linspace(0, H, 50)
    ax.plot(RHO * G * (H - yy), yy, "k-", lw=1.5, label=r"theory $\rho g (H-y)$, H = 0.2 m")
    ax.plot(RHO * G * (Hs - yy), yy, "k--", lw=1, label=rf"theory from surface row $H_s$ = {Hs:.4f} m")
    ax.set_xlabel("pressure [Pa]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"final state: RMS error {rms:.2f} % of $\\rho g H$")
    ax.grid(alpha=0.3)
    ax.legend()

    ax2.plot(tt, pb, "-o", ms=3, label="bottom pressure / theory")
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("p_bottom / theory [-]")
    ax2.set_ylim(0.8, 1.2)
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.semilogy(tv, np.maximum(vmax, 1e-12), "C3-", label="max |u| [m/s]")
    ax3.set_ylabel("max |u| [m/s]", color="C3")
    ax2.legend(loc="upper left")
    ax3.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("compare.png", dpi=150)

    print(f"理論: 底の圧力 rho g H = {p_theory_bottom:.1f} Pa")
    print(f"圧力の RMS 誤差 (rho g H に対する %): 全体 {rms:.2f} %, 壁から離れた所 {rms_in:.2f} %")
    print(f"圧力の傾き dp/dy = {slope:.1f} Pa/m (理論 {-RHO * G:.1f}, 誤差 {(slope / (-RHO * G) - 1) * 100:+.2f} %)")
    print(f"表面の粒子の高さ Hs = {Hs:.4f} m (初期 {H - 0.0025:.4f} m: 体積の変化 {(Hs / (H - 0.0025) - 1) * 100:+.2f} %)")
    print(f"Hs から測った圧力の RMS 誤差: {rms_s:.2f} %  (p = 0 は表面の粒子の中心に課すので、H = 0.2 との差 dp/2 は離散化のずれ)")
    print(f"底の圧力 / 理論 (最後): {pb[-1]:.4f}")
    print(f"最大速度: 最後 {vmax[-1]:.2e} m/s, 全時間の最大 {vmax.max():.2e} m/s"
          f"  (参考: sqrt(g H) = {np.sqrt(G * H):.2f} m/s)")
    print("-> compare.png")


if __name__ == "__main__":
    main()
