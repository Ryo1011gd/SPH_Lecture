#!/usr/bin/env python3
"""box.grid のコンクリート (Type 0) に剛体ラベルを付けて box.rigid を書く。
   1 = 床 (y < 0)、2 = 左側壁 (x < 0)、3 = 右側壁 (x > 0.5)、0 = 剛体ではない
   ラベルごとに rigid_history.csv の残り粒子数が出るので、下向き・横向きの侵食を分けて見られる。"""
labels = []
with open("box.grid") as fp:
    fp.readline(); fp.readline()
    for line in fp:
        v = line.split()
        if not v:
            continue
        t, x, y = int(v[0]), float(v[1]), float(v[2])
        if t != 0:
            labels.append(0)
        elif y < 0.0:
            labels.append(1)
        elif x < 0.0:
            labels.append(2)
        else:
            labels.append(3)
with open("box.rigid", "w") as fp:
    fp.write("# rigid label per grid row: 1 floor, 2 left wall, 3 right wall (concrete, Type 0), 0 none\n")
    fp.write("\n".join(map(str, labels)) + "\n")
print("box.rigid:", {k: labels.count(k) for k in (0, 1, 2, 3)})
