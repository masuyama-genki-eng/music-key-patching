# SPEC.md — TonalWM 実験プロトコル（正確な仕様）

Status: **pre-registration draft v0.1**（実験開始前に凍結し、変更は CHANGELOG に理由付きで記録）
本文書の閾値・判定基準はすべて**実験前に定めた決定規則**であり、結果の予測ではない。

---

## 0. 記法

- 系列 s、トークン位置 t、層 ℓ ∈ {0..L−1}、残差ストリーム活性 h_ℓ(t) ∈ R^d
- key κ = (tonic ∈ Z_12, mode ∈ {maj, min})、24 クラス
- 移調作用 T_k：全ピッチを +k 半音（k ∈ Z_12）

## 1. データ

### 1.1 D-SYN（合成・主データ）

機能和声文法によるジェネレータ。**全トークンに ground-truth key ラベル**。

- 文法：状態機械 T→S→D→T（代理和音: ii↔IV, vi 経由等）。和音は 4 声体（SATB 範囲制約、平行 5/8 度回避は簡略規則）＋任意でメロディ声部（和声音 + 経過音・刺繍音）
- 転調：系列あたり 0–3 回。種類 = {pivot-chord, direct, sequential}、遷移先は五度圏距離 1–6 を層化サンプリング。転調点・新 key を token 単位で記録
- 規模：train 200k / val 10k / test 10k 系列、長さ 256–512 トークン
- パラメータは `configs/data_syn.yaml` に全記載。ジェネレータは seed 固定・決定的

### 1.2 トークナイザ（漏洩防止が最優先）

- 語彙：`BAR`, `POS_{1..16}`, `PITCH_{21..108}`（**絶対 MIDI ピッチ**）, `DUR_{...}`, `VEL_{...}(任意)`, `EOS`
- **禁止**：key / chord / ローマ数字 / scale-degree トークン。調情報は音の並びからのみ復元可能でなければならない
- 実装検査：語彙リストに対する禁止語 unit test（`tests/test_vocab_no_leak.py`）

### 1.3 D-REAL（実データ・secondary）

- 候補：Hooktheory/TheoryTab 系（key + RN 注釈）、When-in-Rome、Bach chorales（music21 の key 解析 = アルゴリズム由来と明記）
- **実装時にライセンスを確認し、利用可否を LEDGER に記録**。利用不可なら D-REAL は Bach chorales のみで縮退
- 用途：probe の生態学的検証のみ（M-CTRL の訓練には D-SYN を主とし、D-REAL 混合は R-Mixed 条件として任意）

## 2. モデル

### 2.1 M-CTRL（自前訓練・主対象）

- GPT-2 型 decoder-only：L=8, heads=8, d=512, context 512（Othello-GPT 準拠）
- 訓練条件（**操作変数**）：
  - **R-NoAug**：移調 augmentation なし
  - **R-Aug**：一様 12 移調 augmentation
- 各条件 seed ≥ 2（計算が許せば 3）。最適化・LR 等は `configs/train.yaml`。val loss 曲線と最終 perplexity を LEDGER に記録
- 品質ゲート：val next-token top-1 が chance を大幅超過し、無条件生成の in-key ratio が D-SYN 統計と乖離しないこと（乖離閾値は訓練後・介入前に val 統計から設定し記録）

### 2.2 M-REF(参照モデル・musicality guard 用)

- M-CTRL と同構成・**別 seed・別データ分割**で訓練。介入評価の perplexity 計測専用。circular evaluation を避けるため M-CTRL と重みを共有しない

### 2.3 M-WILD（Phase C）

- 公開事前学習 symbolic checkpoint（実装時に入手可能性・ライセンス確認）。tokenizer 差異は adapter probe で吸収

## 3. Phase A — 存在と符号化形式（H1, H2）

### A1. Key probing

- 各層 ℓ・各位置 t の h_ℓ(t) から κ(t) を多項ロジスティック回帰（線形）で予測。secondary に 2 層 MLP
- 分割：系列単位で train/val/test。指標：24-class macro-F1（tonic 12-class / mode 2-class も分解報告）
- **統制**：
  - C1 selectivity：ラベルを系列内シャッフルした control task との差
  - C2 untrained：ランダム初期化 M-CTRL への同一 probe
  - C3 入力ベースライン：直近窓 W ∈ {8,16,32,64} トークンの (i) pitch-class histogram → logistic 回帰、(ii) Krumhansl-Schmuckler 相関法。**probe 精度は常に max(C3) と併記**
- **判定規則 DR-H1**：ある層で [selectivity 補正後 probe F1] − [best C3 F1] > 0 が系列単位 bootstrap 95%CI で 0 を含まない場合、H1 支持。全層で含む場合、H1 不支持

### A2. 曖昧度層別

- 各位置の曖昧度 = KS 法の key 事後分布エントロピー（窓 W=16）
- 高曖昧ビン（上位 25%）での probe−C3 差を主要図に。転調点 ±8 トークンの追跡（内部 key 反転のラグ中央値）を副次図に

### A3. 符号化形式（H2）

- **同変性**：ペア {s, T_k s} の活性に対し層ごとに Procrustes で線形写像 R_k を推定。検定量：
  - 巡回性誤差 ε_cyc = mean_k ‖R_k − R_1^k‖_F / ‖R_k‖_F
  - 汎化：R_k を train 系列で推定し held-out 系列の表現整列誤差で評価
- **判定規則 DR-H2b**：R-Aug の ε_cyc < R-NoAug の ε_cyc（seed をまたぐ Wilcoxon、片側）
- 幾何の探索的記述：probe の tonic 12 方向を 2D に射影し、隣接構造が chromatic / fifths のどちらに近いか（円順序の Kendall τ）を報告（判定規則なし・記述のみ）

## 4. Phase B — 因果介入（H3, H5）

### B1. Key 部分空間の構成（3 手法を比較）

1. **V-PROBE**：層 ℓ の probe 重み行列の行空間（rank ≤ 24、実際は直交化して使用）
2. **V-MEAN**：クラス条件付き平均 μ_κ(ℓ) の張る空間。edit は成分置換 h ← h − P_V h + P_V μ_{κ*}(ℓ)
3. **V-DAS**：interchange intervention accuracy を目的関数に学習する低ランク直交部分空間（rank r ∈ {8, 24} を sweep）。目的：donor（key κ′）の部分空間成分を receiver（key κ）に移植した際、継続が κ′ に従う確率を最大化。held-out prompt で評価（ITE の three-stage 検証の流儀）

### B2. 介入プロトコル

- Prompt：test 系列の先頭 8 小節（key κ_src 確定区間）。生成：継続 16 小節、temperature/top-p は `configs/gen.yaml` に固定
- Edit：小節境界 t* 以降の全ステップで部分空間成分を target key κ* に置換（sustained）。one-shot（t* のみ）は Phase C の持続分析用
- 条件：κ* = κ_src から五度圏距離 1–6 の 12 tonic（mode 固定 major を主、minor は副）× prompt n ≥ 100 × 層 ℓ sweep（stride 1、L=8 なので全層）
- **統制**：
  - K1 rank・norm 整合ランダム部分空間 edit
  - K2 sham edit（P_V h を除去して同じものを戻す）→ 出力は clean と bit 一致すること（実装検証を兼ねる）
  - K3 層シャッフル（V を別層の基底で適用）
  - K4 行動的上限参照：prompt を κ* に移調して与えた場合の継続（介入なしの「理想」挙動）

### B3. 評価指標

- **TKR**（target-key realization）：継続の推定 key（KS、継続全窓）が κ* に一致する率。参考 chance = 1/12（tonic）
  - **評価器の測定済み限界（2026-07-11、starter 実装時に発見・記録）**：KS 法は clean な非転調 16 小節に対し正確一致 32/40 = 80%、混同は**全件が五度圏距離 1（属調/下属調系）**だった（生成物は完全ダイアトニックであることを検証済み＝評価器側の限界）。対策：(i) TKR は strict（完全一致）と tolerant（近親調許容：五度圏距離 ≤1・平行・関係調）の両方を報告、(ii) 12 標的の特異性行列で評価器バイアスを可視化、(iii) KS 非依存の IKR を常に併記して三角測量する
- **IKR_target / IKR_src**：継続ピッチの κ* / κ_src ダイアトニック率
- **特異性行列**：12 target × 推定 key の混同行列。五度圏距離 vs 成功率の曲線
- **Musicality guard**：M-REF による継続 perplexity。予算 δ_PPL は「D-SYN val 中の自然転調直後区間の perplexity 上昇分布の 90 パーセンタイル」として**介入実験前に算出し LEDGER に凍結**。ガード超過の edit は成功と数えない
- 文法統計（音域逸脱率、極端な同音連打率）を副次ガードとして併記

### B4. 判定規則

- **DR-H3**：最良層・最良部分空間法において、TKR(edit) − TKR(K1) > 0 が 12 target 中 ≥ 8 で Holm 補正 Wilcoxon p < .05、かつ当該条件がガード予算内。満たせば H3 支持
- **DR-H5**:TKR(edit) の層プロファイルにおいて、最良層の効果量が層中央値の効果量を bootstrap CI で上回る（局在の存在）。プロファイル形状（early/mid/late）は記述報告
- **方向間比較**（explanatory）：V-DAS − V-PROBE の TKR 差 = actionability gap の世界モデル版として報告

## 5. Phase C — 動態と一般化（H4、OJSP 拡張）

- C1 持続：one-shot edit 後の IKR_target(t) 軌跡、半減期、再主張率。遷移の音楽的性質は探索的アノテーション（著者＋研究室メンバー、ブラインド、n 小規模、exploratory と明記）
- C2 第二状態変数：拍節位相（beat-in-bar）。C3 入力ベースラインが強い変数なので、主眼は causal 側（位相 edit → downbeat 再アンカー）
- C3 M-WILD 一般化、C4 R-Aug/NoAug の介入可能性差、C5 D-REAL probe 検証

## 6. 統計一般

- 対応あり比較は系列（prompt）単位でペア化。Wilcoxon signed-rank、多重比較は Holm。効果量 rank-biserial r。CI は BCa bootstrap 10k
- すべての検定は SPEC 記載のものに限る。追加の探索的分析は「exploratory」と明記し補正なしの p を主張に使わない

## 7. 研究公正プロトコル（改竄防止・最重要）

1. **RESULTS_LEDGER.md**：全実行を {datetime, git hash, config hash, seed, 出力 artifact パス} で追記型記録。書き換え禁止（追記のみ）
2. **論文・レポートに書く数値は、results/ 配下の実行生成物に由来するものに限る**。プレースホルダ数値・「期待される値」の記入は禁止。未実行の欄は "TBD (run required)" と書く
3. 負の結果・ガード超過・失敗 run はすべて報告対象。除外には SPEC 記載の事前基準以外を用いない
4. seed は configs に固定・列挙。cherry-picking 検査として、報告図の背後にある全 seed の分布を appendix に必ず出す
5. 分析コードと生成コードを分離（`analysis/` は `results/` のみを読む）
6. SPEC の変更は実験着手後は CHANGELOG.md に日付・理由付きで記録（pre-registration の透明性）

## 8. 計算予算の見積り（拘束条件）

- M-CTRL（~25M params 級）×（2 regimes × 2–3 seeds）：単一 GPU で現実的
- Phase B 生成：12 targets × 100 prompts × 8 layers × {edit, K1} ≈ 19.2k 継続 + 統制少数。16 小節継続 ≈ 数百トークン。バッチ生成で単一 GPU 数日以内を想定。超過する場合は層 stride 2 → 峰の周辺精査に縮退（縮退判断も LEDGER に記録）

## 9. 投稿計画

- **ICASSP 2027**：締切 2026-09-16 (AoE)。内容 = Phase A（H1, H2 core）+ Phase B（H3, H5）。M-CTRL / R-Aug 主線
- **OJSP regular**（rolling、2027 Q1 目標）：+ Phase C 全部、R-NoAug 比較、M-WILD、metric-position
- 週次マイルストーン：
  - W1–2（–7/26）：データ生成器・tokenizer・訓練開始
  - W3–4（–8/9）：Phase A（引越しバッファ込み）
  - W5–6（–8/23）：Phase B core
  - W7–8（–9/6）：解析・図・執筆（KNOWLEDGE §7 の執筆規範を適用）
  - W9（–9/16）：内部レビュー（櫻井先生）→ 提出
