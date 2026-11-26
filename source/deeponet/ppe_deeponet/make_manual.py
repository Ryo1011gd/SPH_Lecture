"""Generate PPE_DeepONet_manual.pdf (Japanese user manual).

    python3 make_manual.py            -> PPE_DeepONet_manual.pdf next to this file
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
GOTHIC, MINCHO = "HeiseiKakuGo-W5", "HeiseiMin-W3"

# ----------------------------------------------------------------------------
# measured results (filled from the benchmark runs)
# ----------------------------------------------------------------------------
RESULTS = [
    # case, n, Jacobi iter, DeepONet iter, PPE Jacobi [s], PPE DeepONet [s], wall Jacobi [s], wall DeepONet [s]
    ("D01 粘性崩壊 AR2.8", 9511, 125.0, 2.00, 13.03, 0.531, 16.9, 4.01),
    ("D03 斜面 20°", 9562, 124.4, 2.01, 12.73, 0.561, 17.2, 4.62),
    ("D04 Jeffreys 薄膜流", 1181, 37.6, 1.00, 3.78, 0.347, 6.6, 2.90),
    ("V05 Taylor-Green (閉じた系)", 10000, 57.1, 1.03, 5.90, 0.444, 10.3, 4.99),
    ("V03 Spin-up (閉じた系)", 9692, 55.0, 1.10, 5.71, 0.487, 10.3, 5.20),
]
RESULT_NOTE = ("RTX 6000 Ada, nvc++ 24.1, 300 ステップ, tol 1e-8, PpeDonChebDegree 10 / ChebRatio 30 / SetupEvery 10。"
               "5 ケースとも学習データに含まれていない (未学習)。フォールバック 0 回。"
               "PPE 時間は DeepONet の推論・Galerkin 粗行列の構築・λmax 推定・真の残差チェックをすべて含む。"
               "最終状態は Jacobi 版と一致 (粒子位置の差 5e-5 dp 以下、速度の相対差 6e-7 以下、two-phase の診断値は 7 桁一致)。")

# ----------------------------------------------------------------------------
H1 = ParagraphStyle("h1", fontName=GOTHIC, fontSize=15, leading=20, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b6f"))
H2 = ParagraphStyle("h2", fontName=GOTHIC, fontSize=12, leading=16, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1f3b6f"))
BODY = ParagraphStyle("body", fontName=MINCHO, fontSize=9.5, leading=15, alignment=TA_LEFT, spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CELL = ParagraphStyle("cell", fontName=MINCHO, fontSize=8.5, leading=11)
CELLB = ParagraphStyle("cellb", fontName=GOTHIC, fontSize=8.5, leading=11)
CODE = ParagraphStyle("code", fontName=GOTHIC, fontSize=7.8, leading=10.5, backColor=colors.HexColor("#f3f4f6"),
                      borderPadding=(4, 4, 4, 4), leftIndent=4, rightIndent=4, spaceBefore=3, spaceAfter=7)
TITLE = ParagraphStyle("title", fontName=GOTHIC, fontSize=20, leading=28, spaceAfter=6)
SUB = ParagraphStyle("sub", fontName=MINCHO, fontSize=11, leading=16, textColor=colors.HexColor("#444444"))


def p(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def code(t):
    return Preformatted(t.strip("\n"), CODE)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa4b2")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eaf3")))
    t.setStyle(TableStyle(st))
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont(MINCHO, 8)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(18 * mm, 10 * mm, "ISPH + DeepONet PPE 前処理 利用マニュアル")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def build(path):
    S = []
    S += [Spacer(1, 30 * mm), p("ISPH 圧力ポアソン方程式の<br/>DeepONet 前処理 利用マニュアル", TITLE),
          p("main_implicit.cpp / ppe_deeponet.inc / ppe_deeponet/*.py", SUB), Spacer(1, 6 * mm),
          p("教師データ作成 (make DATASET=1) → 学習 (python3 / make train) → 前処理付き計算 (make ONET=1)", SUB),
          Spacer(1, 14 * mm)]
    S.append(table([
        ["項目", "内容"],
        ["対象", "main_implicit.cpp の PPE (BiCGStab, 収束判定 |r|/|b| < PpeTolerance = 1e-8)"],
        ["前処理", "Chebyshev 多項式平滑化 + DeepONet 粗空間 (Galerkin) の 2 レベル前処理"],
        ["精度", "収束判定は従来と同一。解は Jacobi 版と許容誤差の範囲で一致する"],
        ["安全性", "演算子・推論の自己検証、真の残差での最終判定、失敗時は従来の Jacobi BiCGStab に自動切替"],
        ["対応ビルド", "GPU (nvc++ OpenACC) / CPU (g++ OpenMP)、2 次元 (DIM=2) / 3 次元 (DIM=3)"],
    ], [32 * mm, 138 * mm]))
    S.append(PageBreak())

    # 1 ------------------------------------------------------------------
    S.append(p("1. 全体の流れ", H1))
    S.append(p("作業は 3 段階です。参考にした DeepONet-SPH と同じく、make のオプションで実行ファイルを切り替えます。"))
    S.append(code("""
# (0) nvc++ に PATH を通す (例)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/bin:$PATH
cd /media/ryo/NewVolume/Implicit_SPH/source

# (1) 教師データ作成用の実行ファイル -> SPH_acc_dataset
make GPU=1 DATASET=1
#     各ケースで通常どおり実行すると ppe_don_data/ に PPE スナップショットが出力される

# (2) 学習 -> ppe_deeponet/trained_model/ppe_deeponet.bin
make train DATA="caseA/ppe_don_data caseB/ppe_don_data ..."
#   (= python3 ppe_deeponet/train_ppe_don.py --data ... --out ppe_deeponet/trained_model)

# (3) DeepONet 前処理付きの実行ファイル -> SPH_acc_onet
make GPU=1 ONET=1
#     .data ファイルを変更せずにそのまま実行できる
"""))
    S.append(table([
        ["make コマンド", "実行ファイル", "動作"],
        ["make GPU=1", "SPH_acc", "従来どおり (Jacobi BiCGStab)"],
        ["make GPU=1 DATASET=1", "SPH_acc_dataset", "従来の計算 + 50 ソルブごとに PPE スナップショットを出力"],
        ["make GPU=1 ONET=1", "SPH_acc_onet", "DeepONet 前処理を既定で有効化 (モデルのパスを埋め込む)"],
        ["make CPU=1 [DATASET=1|ONET=1]", "SPH_omp[_dataset|_onet]", "g++ + OpenMP 版"],
        ["make train DATA=\"...\"", "-", "学習を実行し ppe_deeponet/trained_model/ に保存"],
        ["既存ターゲット (make SPH_acc 等)", "従来どおり", "変更なし"],
    ], [52 * mm, 38 * mm, 80 * mm]))
    S.append(Spacer(1, 3 * mm))
    S.append(table([
        ["make 変数", "既定値", "意味"],
        ["DIM", "2", "2: 2 次元 (-D TWO_DIMENSIONAL)、3: 3 次元。モデルの次元と一致させる"],
        ["GPU_ARCH", "cc89", "nvc++ の -gpu= に渡す GPU アーキテクチャ"],
        ["EVERY", "50", "DATASET=1 のときスナップショットを出力する間隔 (PPE ソルブ数)"],
        ["MODEL", "ppe_deeponet/trained_model/ppe_deeponet.bin", "ONET=1 で埋め込むモデルの絶対パス"],
        ["NVCXX / PYTHON", "nvc++ / python3", "コンパイラ・Python の実行ファイル"],
        ["EPOCHS", "30", "make train の学習エポック数"],
    ], [28 * mm, 62 * mm, 80 * mm]))
    S.append(p("DATASET=1 と ONET=1 は同時に指定できません。CPU=1 と GPU=1 も同様です。"
               "どの実行ファイルでも、下記の .data キーを書けば既定値を上書きできます。"))

    # 2 ------------------------------------------------------------------
    S.append(p("2. ステップ 1: 教師データの作成", H1))
    S.append(p("SPH_acc_dataset は通常の Jacobi BiCGStab で計算し、収束した PPE を一定間隔でファイルに書き出します。"
               "各ファイルには、粒子特徴量・行列 A (CSR)・右辺 b・収束解 p・warm start p0 が含まれます。"))
    S.append(code("""
cd <ケースのディレクトリ>
<source>/SPH_acc_dataset box.data box.grid box%03d.prof box%03d.vtk box.log 4
ls ppe_don_data/          # ppe_0000050.bin, ppe_0000100.bin, ...  (約 5 MB/枚, 1 万粒子)

# 出力の確認: |b - A p|/|b| が許容誤差程度になっていれば正しく出力されている
python3 <source>/ppe_deeponet/ppe_don_data.py ppe_don_data
"""))
    S.append(p("複数ケースを一括で作る場合は、付属スクリプトを使えます (EndTime と出力間隔を上書きして実行します)。"))
    S.append(code("""
ppe_deeponet/gen_snapshots.sh ./SPH_acc data 0.5 50 ../results/.../cases/D01_* ../results/.../cases/V01_*
#                              実行ファイル 出力先 EndTime 間隔 ケースのディレクトリ...
"""))
    S.append(p("データ作成の指針", H2))
    S += bullets([
        "形状・粘性・解像度・自由表面の形が異なるケースを複数用意する (今回の学習は 8 ケース × 100 枚)。",
        "閉じた容器 (自由表面なし) や two-phase のケースを使う場合は、それらも学習データに含める。",
        "検証は時刻ではなく<b>ケース単位で</b>分けて行う (学習ではケースごと除外して検証する)。",
        "スナップショットの出力に計算時間はほとんどかからない (1 枚あたり数十 ms)。",
    ])

    # 3 ------------------------------------------------------------------
    S.append(p("3. ステップ 2: 学習", H1))
    S.append(code("""
python3 ppe_deeponet/train_ppe_don.py \\
    --data caseA/ppe_don_data caseB/ppe_don_data caseC/ppe_don_data ... \\
    --val-data caseH/ppe_don_data \\
    --out ppe_deeponet/trained_model --epochs 30 --fourier-max 10
"""))
    S.append(p("必要なパッケージ: torch, numpy, scipy (GPU があれば自動的に使用)。800 枚・30 エポックで RTX 6000 Ada で約 20 分です。"))
    S.append(table([
        ["主なオプション", "既定値", "意味"],
        ["--data", "(必須)", "スナップショットのディレクトリ (複数可)"],
        ["--val-data", "なし", "検証用ケース。省略時はケースの 20 % を丸ごと検証用に除外"],
        ["--basis", "64", "DeepONet の基底数 P (粗空間の次元、最大 128)"],
        ["--cheb-degree / --cheb-ratio", "10 / 30", "学習時に想定する Chebyshev 平滑化 (配布モデルは 2 次で学習し、実行時 10 次で最良)"],
        ["--epochs / --lr", "60 / 2e-3", "エポック数 / 学習率 (OneCycle)"],
        ["--queries", "2048", "教師あり項で trunk を評価するランダムな粒子数"],
        ["--w-galerkin / --w-learned / --w-solution", "1 / 1 / 0.3", "各損失の重み"],
    ], [52 * mm, 24 * mm, 94 * mm]))
    S.append(p("出力ファイル (--out のディレクトリ)", H2))
    S.append(table([
        ["ファイル", "内容"],
        ["ppe_deeponet.bin", "C++ が読み込む重み (検証損失が最良のエポック)"],
        ["ppe_deeponet.data.txt", "このモデルを学習したときの設定。.data に貼り付けて使う"],
        ["best.pt / last.pt", "PyTorch のチェックポイント (評価・追加学習用)"],
        ["history.json", "エポックごとの損失・検証時の反復回数"],
    ], [45 * mm, 125 * mm]))
    S.append(p("学習ログの読み方", H2))
    S.append(code("""
ep  24  ... | val galerkin=3.3e-03 ... iter_jacobi=123.3 iter_galerkin=2.4 iter_learned=9.7
"""))
    S += bullets([
        "galerkin: 2 レベル法の誤差縮小率の 2 乗 (小さいほど良い)。",
        "iter_jacobi / iter_galerkin: 検証ケースで BiCGStab を実際に解いたときの反復回数 (torch で C++ を忠実に再現)。",
        "iter_learned: ネットワークだけで粗補正を行う方式 (PpeDonCoarse 0) の反復回数 (参考値)。",
    ])
    S.append(p("オフライン評価 (学習に使っていないケースで確認)", H2))
    S.append(code("""
python3 ppe_deeponet/eval_ppe_don.py --model ppe_deeponet/trained_model/best.pt \\
    --data caseX/ppe_don_data caseY/ppe_don_data --modes jacobi galerkin
"""))

    # 4 ------------------------------------------------------------------
    S.append(p("4. ステップ 3: DeepONet 前処理付きの計算", H1))
    S.append(code("""
make GPU=1 ONET=1                       # MODEL=... で別のモデルも指定できる
cd <ケースのディレクトリ>
<source>/SPH_acc_onet box.data box.grid box%03d.prof box%03d.vtk box.log 4
"""))
    S.append(p("SPH_acc_onet は前処理が既定で有効です。通常の実行ファイル (SPH_acc) でも、.data に次の行を書けば同じ動作になります。"))
    S.append(code("""
PpeDon              1
PpeDonModel         /media/ryo/NewVolume/Implicit_SPH/source/ppe_deeponet/trained_model/ppe_deeponet.bin
"""))
    S.append(p(".data キー一覧 (すべて省略可)", H2))
    S.append(table([
        ["キー", "既定値", "意味"],
        ["PpeDon", "0 (ONET 版は 1)", "1: DeepONet 前処理を使う"],
        ["PpeDonModel", "ppe_deeponet.bin (ONET 版は埋め込みパス)", "モデルファイル"],
        ["PpeDonCoarse", "1", "1: DeepONet 基底 + Galerkin (推奨)、0: ネットワークのみの粗補正"],
        ["PpeDonChebDegree", "10", "Chebyshev 平滑化の次数 (0 にすると減衰 Jacobi)"],
        ["PpeDonChebRatio", "30", "Chebyshev の区間 [λmax/比, λmax]"],
        ["PpeDonPreSmooth / PostSmooth", "1 / 1", "前・後平滑化の有無"],
        ["PpeDonInitialGuess", "1", "x0 ← x0 + M<super>-1</super> r0 (残差が下がる場合のみ採用)"],
        ["PpeDonSetupEvery", "1", "基底と粗行列を作り直す間隔。10 程度にすると高速 (精度は不変)"],
        ["PpeDonLmaxEvery", "20", "λmax(D<super>-1</super>A) を推定し直す間隔"],
        ["PpeDonVerifyEvery", "500", "格納した演算子を applyPpeOperator と照合する間隔"],
        ["PpeDonExportEvery / ExportDir", "0 (DATASET 版は 50) / ppe_don_data", "スナップショットの出力"],
    ], [42 * mm, 46 * mm, 82 * mm]))
    S.append(p("ログの見方", H2))
    S.append(code("""
PPE-DeepONet: device inference self-test on 38 particles: ... -> ok      (初回のみ)
PPE-DeepONet: iter=2 rel=6.1e-09 setup=0.4ms solve=1.9ms lmax=1.33 | mean iter=2.00 fallback=0
ISPH(...): iter=2 rel=6.1e-09 ...                                         (従来と同じ形式)
PPE summary: DeepONet BiCGStab: runs=301 mean iter=2.00 fallback=0 ... total=0.53s
PPE summary: DeepONet time per solve [ms]: operator=... trunk=... precond-apply=...
PPE summary: Jacobi time per solve [ms]: ...                              (フォールバック時)
"""))
    S.append(p("fallback が増える、self-test が FAILED、stored-operator check が MISMATCH と表示される場合は、"
               "第 7 章を参照してください。いずれの場合も計算は従来の Jacobi BiCGStab で継続されます。"))

    # 5 ------------------------------------------------------------------
    S.append(p("5. 性能", H1))
    if RESULTS:
        rows = [["ケース (すべて未学習)", "粒子数", "反復 Jacobi → DeepONet", "PPE 時間 [s]", "全体 [s]"]]
        for r in RESULTS:
            rows.append([r[0], r[1], f"{r[2]:.1f} → <b>{r[3]:.2f}</b>", f"{r[4]:.2f} → {r[5]:.2f} ({r[4]/r[5]:.1f}倍)",
                         f"{r[6]:.1f} → {r[7]:.1f} ({r[6]/r[7]:.1f}倍)"])
        S.append(table(rows, [46 * mm, 15 * mm, 34 * mm, 38 * mm, 37 * mm]))
        S.append(p(RESULT_NOTE))
    S.append(p("Chebyshev の次数と反復・時間の関係 (D01 AR2.8、300 ステップ)", H2))
    S.append(table([
        ["PpeDonChebDegree", "平均反復", "PPE 時間合計", "全体"],
        ["2", "6.84", "0.750 s", "4.25 s"],
        ["4", "3.98", "0.560 s", "4.04 s"],
        ["6", "3.00", "0.536 s", "4.04 s"],
        ["8", "2.49", "0.529 s", "4.00 s"],
        ["10 (既定)", "2.00", "0.531 s", "4.01 s"],
        ["12", "2.00", "0.548 s", "3.99 s"],
    ], [40 * mm, 30 * mm, 40 * mm, 30 * mm]))
    S.append(p("次数を上げると 1 回の前処理は重くなりますが、反復ごとのホスト同期が減るため、合計時間はむしろ短くなります。"
               "10 次を既定値としています (D03 斜面 20° では 8 次: 2.93 回、10 次: 2.01 回)。"))

    # 6 ------------------------------------------------------------------
    S.append(p("6. 手法の概要", H1))
    S.append(code("""
M^-1 r :  z  = Cheb_k(r; 0)                     前平滑化 (D^-1 A のスペクトル [lmax/30, lmax] を除去)
          z += T (T^T D^-1 A T)^-1 T^T D^-1 (r - A z)   DeepONet 粗補正 (現在の A で Galerkin)
          z  = Cheb_k(r; z)                     後平滑化
"""))
    S += bullets([
        "<b>粒子法向けの DeepONet</b>: センサー格子を使わない。trunk は粒子位置 (正規化座標 + 多重スケール Fourier 特徴 + 局所幾何特徴) で評価し、"
        "形状の潜在変数は粒子集合の平均プーリング (並べ替え不変) で求める。学習時は trunk をランダムな粒子で評価する。",
        "<b>残差に対して線形</b>: 残差は非線形関数を通らないので、1 回のソルブ中は固定の線形作用素になり、通常の BiCGStab がそのまま使える。"
        "残差の大きさについての外挿問題も生じない。",
        "<b>Galerkin 粗空間 (外挿に強い)</b>: 粗行列は毎回その時点の A から組み立てる。ネットワークは部分空間だけを担当するので、"
        "初めての形状でも反復が少し増えるだけで、解が誤ることはない。",
        "<b>スペクトル分担</b>: D<super>-1</super>A の固有値は (0, 約 1.45] の正の実数。Chebyshev で 0.05 以上を除去し、"
        "残る数十個の大域的な低固有値モードを DeepONet の基底 (P=64) が担当する。",
        "<b>教師なし学習</b>: 出力した行列 A を使い、2 レベル法の誤差伝播 ||S(I-CA)S e|| を、白色雑音とゆっくり変化する探索ベクトルで評価して最小化する。"
        "PPE を追加で解く必要はない。",
        "<b>閉じた系</b>: A·1 = 0 を自動検出し、基底と前処理出力から定数 (圧力のゲージ自由度) を除去する。",
    ])

    # 7 ------------------------------------------------------------------
    S.append(p("7. トラブルシューティング", H1))
    S.append(table([
        ["症状", "原因と対処"],
        ["cannot open model ... -> Jacobi BiCGStab only", "モデルのパスが誤っている。PpeDonModel を絶対パスで指定するか、make GPU=1 ONET=1 MODEL=... で作り直す。"],
        ["model dimensions ... do not match", "2 次元モデルを 3 次元ビルドで読んでいる (またはその逆)。DIM を揃えて学習・ビルドする。"],
        ["stored-operator check ... MISMATCH", "applyPpeOperator() を変更したが ppe_deeponet.inc の donBuildOperator() を合わせていない。両方を同じ式にする。"],
        ["self-test ... FAILED", "デバイス上の推論結果がホストでの計算と一致しない (コンパイラの問題など)。DeepONet は自動的に無効化される。nvc++ のバージョンを変えて確認する。"],
        ["fallback が頻発し、時間が増える", "学習データと大きく異なる流れ。そのケースを DATASET=1 で出力して学習データに加え、再学習する。"],
        ["反復は減るが速くならない", "粒子数が少ないと、カーネル起動とホスト同期の待ちが支配的になる。PpeDonSetupEvery 10 を指定し、PpeDonChebDegree を 4〜8 で調整する。"],
        ["nvc++: command not found", "export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/bin:$PATH、または make NVCXX=... を指定する。"],
    ], [58 * mm, 112 * mm]))
    S.append(Spacer(1, 3 * mm))
    S.append(p("nvc++ 23.x / 24.1 では、スレッドごとのローカル配列・内側のリダクション・ストライドのある書き込みを組み合わせた形のカーネルで、"
               "書き込みが黙って消えるという不具合を確認しました (-O1 以上)。ppe_deeponet.inc ではこの形を避けており、"
               "起動時の self-test で検出できます。"))

    # 8 ------------------------------------------------------------------
    S.append(p("8. ファイル一覧", H1))
    S.append(table([
        ["ファイル", "役割"],
        ["main_implicit.cpp", "ソルバ本体 (#include \"ppe_deeponet.inc\" と calculatePpe 内に数行のフックを追加)"],
        ["ppe_deeponet.inc", "C++/OpenACC 実装: 特徴量・推論・Galerkin 粗行列・Chebyshev・BiCGStab・スナップショット出力"],
        ["makefile", "CPU/GPU × DATASET/ONET の切り替えと make train"],
        ["ppe_deeponet/ppe_don_data.py", "スナップショットの読み込みと確認"],
        ["ppe_deeponet/ppe_don_model.py", "ネットワーク・平滑化・2 レベルサイクル・重みの出力"],
        ["ppe_deeponet/ppe_don_solver.py", "C++ の BiCGStab と前処理を torch で忠実に再現したもの"],
        ["ppe_deeponet/train_ppe_don.py / eval_ppe_don.py", "学習 / オフライン評価"],
        ["ppe_deeponet/gen_snapshots.sh", "複数ケースのスナップショットを一括作成"],
        ["ppe_deeponet/trained_model/", "学習済みモデル (8 ケース × 100 スナップショット、P=64)"],
        ["*.pre_deeponet_20261013", "変更前の main_implicit.cpp と makefile"],
    ], [62 * mm, 108 * mm]))

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="ISPH DeepONet PPE preconditioner manual")
    doc.build(S, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PPE_DeepONet_manual.pdf")
    build(out)
    print(out)
