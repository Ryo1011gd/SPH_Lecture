#!/usr/bin/env python3
"""演習2: CCI-2 体系の MCCI (簡略) — コンクリート侵食深さを CCI-2 実験と比較し、エネルギー収支を確認する。

使い方:
    ./compare.sh                          # このフォルダの結果 (rigid_history.csv, output/)
    ./compare.sh ref k3000 k12000         # 感度解析: ケースのフォルダを並べる
      (フォルダをコピーして thermal.data を変え、generate.sh と execute.sh を実行しておく)
実験値: cci2_exp.csv (Farmer et al., OECD/MCCI-2004-TR05, Fig. 4-18 を目視で読み取った局所の侵食深さ)
時間: 伝熱を S 倍速にしているので 実時間 = S x 計算時間。S は thermal.data の Conductivity (Type0) / 1.5 で求める
出力: compare.png, snap.png (温度分布), ablation.csv
  上左: 床 (下向き) の平均侵食深さ と 実験 (床の中心)
  上右: 側壁 (横向き) の平均侵食深さ と 実験 (北・南の側壁、高さ +10 cm)
  下左: 溶融物の平均温度
  下右: 溶融物が出した熱 (発熱を含む) とコンクリートが得た熱 (断熱なら一致する = 保存の確認)
必要なもの: Python3, numpy, matplotlib
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

K_REAL_CONCRETE = 1.5   # 実際の LCS コンクリートの熱伝導率 [W/(m K)] (thermal.data の説明)
DP = 0.01           # 粒子間隔 [m]
FLOOR_LEN = 0.50    # 溶融物が接する床の長さ [m]
WALL_LEN = 0.25     # 溶融物が接する側壁の高さ [m] (初期の液面まで)
RHO = {0: 2300.0, 1: 6000.0}   # box.data の Density


def read_thermal(case):
    """thermal.data から Type0/Type1 の物性を読む (感度解析で値を変えても追従する)。"""
    p = {}
    with open(os.path.join(case, "thermal.data")) as fp:
        for line in fp:
            v = line.split("#")[0].split()
            if len(v) >= 3:
                try:
                    p[v[0]] = [float(x) for x in v[1:]]
                except ValueError:
                    pass
    return p


def speedup(p):
    """伝熱の倍速 S = (thermal.data のコンクリートの熱伝導率) / (実際の値)。"""
    return p["Conductivity"][0] / K_REAL_CONCRETE


def read_exp():
    """cci2_exp.csv: series, t_min, depth_cm (先頭 2 行は説明)。"""
    e = np.genfromtxt("cci2_exp.csv", delimiter=",", skip_header=2, names=True, dtype=None, encoding=None)
    return {s: (e["t_min"][e["series"] == s], e["depth_cm"][e["series"] == s]) for s in np.unique(e["series"])}


def h_of_T(T, cp, Ts, Tl, L):
    """ソルバと同じエンタルピ (固相線で 0、固相線と液相線の間で潜熱を線形に放出)。"""
    hl = cp * (Tl - Ts) + L
    if T <= Ts:
        return cp * (T - Ts)
    if T >= Tl:
        return hl + cp * (T - Tl)
    return (T - Ts) / (Tl - Ts) * hl


def energy_bound(p):
    """溶融物の熱をすべてコンクリートを溶かすのに使ったときの侵食面積の上限 [m2/m] (発熱の分は main で足す)。
       溶融物はコンクリートが外れる温度 T_rel より下には冷えられない、とする。"""
    g = lambda k, i: p[k][i]
    cpc, Tsc, Tlc, Lc, fc = g("HeatCapacity", 0), g("SolidusTemperature", 0), g("LiquidusTemperature", 0), \
        g("LatentHeat", 0), g("CriticalSolidFraction", 0)
    cpm, Tsm, Tlm, Lm = g("HeatCapacity", 1), g("SolidusTemperature", 1), g("LiquidusTemperature", 1), g("LatentHeat", 1)
    hl_c = cpc * (Tlc - Tsc) + Lc
    h_rel = (1.0 - fc) * hl_c                                  # 外れるときのエンタルピ
    T_rel = Tsc + (h_rel / hl_c) * (Tlc - Tsc)
    dh_c = h_rel - h_of_T(g("InitialTemperature", 0), cpc, Tsc, Tlc, Lc)   # 300 K から外れるまで [J/kg]
    dh_m = h_of_T(g("InitialTemperature", 1), cpm, Tsm, Tlm, Lm) - h_of_T(T_rel, cpm, Tsm, Tlm, Lm)
    A_melt = FLOOR_LEN * WALL_LEN
    return A_melt * RHO[1] * dh_m / (RHO[0] * dh_c), dh_c


def read_case(case):
    a = np.genfromtxt(os.path.join(case, "rigid_history.csv"), delimiter=",", names=True)
    out = {}
    for lab in (1, 2, 3):
        r = a[a["label"] == lab]
        out[lab] = (r["time"], (r["n0"] - r["n_rigid"]) * DP * DP)   # 侵食された面積 [m2/m]
    # prof: type x y z u v w 温度 エンタルピ
    t, Tm, Hc, Hm, Mm = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(case, "output", "*.prof"))):
        with open(f) as fp:
            time = float(fp.readline())
            fp.readline()
            d = np.loadtxt(fp, ndmin=2)
        if d.shape[1] < 9:
            continue
        m = np.where(d[:, 0] == 0, RHO[0], np.where(d[:, 0] == 1, RHO[1], 0.0)) * DP * DP
        t.append(time)
        Tm.append(d[d[:, 0] == 1, 7].mean())
        Hc.append(np.sum((m * d[:, 8])[d[:, 0] == 0]))
        Hm.append(np.sum((m * d[:, 8])[d[:, 0] == 1]))
        Mm.append(np.sum(m[d[:, 0] == 1]))
        lost = np.sum((d[:, 0] < 2) & ((d[:, 1] < -0.2) | (d[:, 1] > 0.7) | (d[:, 2] < -0.3) | (d[:, 2] > 0.6)))
        if lost:
            print(f"  警告 {case}: t = {time:.1f} s で {lost} 個の粒子が領域の外にいる (漏れ・発散。Dt を下げる)")
    o = np.argsort(t)
    return out, np.array(t)[o], np.array(Tm)[o], np.array(Hc)[o], np.array(Hm)[o], np.array(Mm)[o]


def snapshots(case, name, S):
    """初期・途中・最後の粒子の温度分布 (コンクリートの侵食形状を見る)。"""
    fs = sorted(glob.glob(os.path.join(case, "output", "*.prof")),
                key=lambda f: float(open(f).readline()))
    if not fs:
        return
    pick = [fs[0], fs[len(fs) // 3], fs[2 * len(fs) // 3], fs[-1]]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, f in zip(axs, pick):
        with open(f) as fp:
            t = float(fp.readline())
            fp.readline()
            d = np.loadtxt(fp, ndmin=2)
        sc = ax.scatter(d[:, 1], d[:, 2], c=d[:, 7], s=4, cmap="inferno", vmin=300, vmax=2200)
        m = d[:, 0] == 1
        ax.scatter(d[m, 1], d[m, 2], s=4, facecolors="none", edgecolors="c", linewidths=0.3)
        ax.set_xlim(-0.2, 0.7)
        ax.set_ylim(-0.3, 0.45)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t:.1f} s (real {t * S / 3600:.2f} h)")
    fig.colorbar(sc, ax=axs, label="T [K]  (cyan ring = corium)")
    fig.savefig(name, dpi=120)
    plt.close(fig)


def main():
    cases = sys.argv[1:] or ["."]
    exp = read_exp()
    fig, ((ax, axw), (ax2, ax3)) = plt.subplots(2, 2, figsize=(13, 9))
    for a, ser, mk, lab in ((ax, "axial_centerline", "o", "CCI-2 exp: floor centerline"),
                            (axw, "radial_north_+10cm", "s", "CCI-2 exp: N wall (+10 cm)"),
                            (axw, "radial_south_+10cm", "^", "CCI-2 exp: S wall (+10 cm)")):
        a.plot(exp[ser][0], exp[ser][1], mk + ":", color="k", ms=5, label=lab)
    print(f"{'ケース':20s} {'S':>5s} {'床[cm]':>8s} {'側壁[cm]':>9s} {'横/下':>6s} {'侵食面積/上限':>14s} {'熱収支の誤差':>14s}")
    table = []
    tmax = 0.0
    for c in cases:
        p = read_thermal(c)
        S = speedup(p)
        Amax, dh_c = energy_bound(p)
        ab, t, Tm, Hc, Hm, Mm = read_case(c)
        q = p.get("ThermalHeatSource", [0.0, 0.0])[1]   # 溶融物の発熱 [W/kg] (計算時間基準)
        qin = q * Mm[0] * t                              # 発熱で入った熱 [J/m]
        gain, loss = Hc - Hc[0], Hm[0] - Hm + qin  # コンクリートが得た熱、溶融物が出した熱 (発熱を含む) [J/m]
        Amax += qin[-1] / (RHO[0] * dh_c)               # 発熱で入った熱もすべて侵食に使った場合
        tf, af = ab[1]
        tw = ab[2][0]
        aw = 0.5 * (ab[2][1] + ab[3][1])                  # 左右の平均
        depth_floor = af / FLOOR_LEN * 100
        depth_wall = aw / WALL_LEN * 100
        tmin = tf * S / 60.0                             # 実時間 [min]
        tmax = max(tmax, tmin[-1])
        ax.plot(tmin, depth_floor, "-", lw=2, label=f"SPH {c} (mean)")
        axw.plot(tw * S / 60.0, depth_wall, "-", lw=2, label=f"SPH {c} (mean)")
        ax2.plot(t * S / 60.0, Tm, "-", label=c)
        ax3.plot(t * S / 60.0, loss / 1e6, "-", label=f"{c}: melt out (incl. heating)")
        ax3.plot(t * S / 60.0, gain / 1e6, "--", label=f"{c}: concrete gain")
        table.append((c, tmin, depth_floor, depth_wall))
        Atot = af[-1] + ab[2][1][-1] + ab[3][1][-1]
        herr = (gain[-1] - loss[-1]) / loss[-1] * 100 if loss[-1] > 0 else np.nan
        ratio = depth_wall[-1] / depth_floor[-1] if depth_floor[-1] > 0 else np.nan
        print(f"{c:20s} {S:5.0f} {depth_floor[-1]:8.1f} {depth_wall[-1]:9.1f} {ratio:6.2f} "
              f"{Atot:.4f}/{Amax:.4f} m2 {herr:12.2f} %")
        np.savetxt("ablation.csv" if len(cases) == 1 else f"ablation_{os.path.basename(os.path.abspath(c))}.csv",
                   np.column_stack([tf, tf * S, depth_floor, depth_wall]), delimiter=",",
                   header="t_sim[s],t_real[s],floor_depth[cm],wall_depth[cm]", comments="")
    for a, ttl in ((ax, "floor (axial)"), (axw, "side wall (radial)")):
        a.set_xlim(0, 1.05 * tmax)
        a.set_ylim(0, None)
        a.set_title(ttl)
        a.set_xlabel("real time after melt contact = S x t [min]")
        a.set_ylabel("ablation depth [cm]")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax2.set_xlabel("real time [min]")
    ax2.set_ylabel("mean melt temperature [K]")
    ax2.grid(alpha=0.3)
    ax3.set_xlabel("real time [min]")
    ax3.set_ylabel("heat [MJ per m depth]")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("compare.png", dpi=150)
    # 実験との比較表 (実験は局所の値、計算は平均の値であることに注意)
    grid = [t for t in (15, 30, 60, 90, 120, 180, 240) if t < tmax] + [tmax]
    print("\n侵食深さ [cm]  (時刻 = " + ", ".join(f"{t:.0f}" for t in grid) + " min)")
    for ser, lab in (("axial_centerline", "実験 床 (中心)"), ("radial_north_+10cm", "実験 北側壁 +10cm"),
                     ("radial_south_+10cm", "実験 南側壁 +10cm")):
        te, de = exp[ser]
        print(f"  {lab:22s}" + "".join(f"{np.interp(t, te[1:], de[1:]):7.1f}" for t in grid))
    for c, tmin, df, dw in table:
        print(f"  {'計算 ' + c + ' 床':22s}" + "".join(f"{np.interp(t, tmin, df):7.1f}" for t in grid))
        print(f"  {'計算 ' + c + ' 側壁':22s}" + "".join(f"{np.interp(t, tmin, dw):7.1f}" for t in grid))
    for c in cases:
        snapshots(c, "snap.png" if len(cases) == 1 else f"snap_{os.path.basename(os.path.abspath(c))}.png",
                  speedup(read_thermal(c)))
    print("-> compare.png, snap*.png")


if __name__ == "__main__":
    main()
