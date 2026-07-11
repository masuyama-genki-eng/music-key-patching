# KNOWLEDGE.md — Tonal World Models（背景・関連研究・仮説・設計根拠）

Project: **TonalWM — Does a Music Transformer Know What Key It Is In?**
Companion docs: `SPEC.md`（正確なプロトコル）, `CLAUDE.md`（実装指示）
Last updated: 2026-07-11

---

## 1. 中心的な問い

音楽生成 Transformer は、**現在の調（key = tonic pitch class × mode）という潜在状態**を内部に保持し、それを生成に **causal に使用している**か？ それとも表層統計（直近のピッチ分布）の連鎖で調性らしさを実現しているだけか？

これは Othello-GPT (Li et al. 2023) が盤面状態について立てた問いの、音楽への正確な写像である。音楽における「盤面」は調・拍節位置・和声機能であり、その第一変数として key を選ぶ。

## 2. 方法論の系譜（なぜこの型が信頼できるか）

世界モデル介入のパラダイムは以下で確立・反復済み：

- **Othello-GPT** (Li et al. 2023)：棋譜のみで訓練した GPT に盤面の内部表現を発見。表現への介入が以降の合法手予測を書き換えた盤面に整合させることを示した。
- **Nanda (2023) の再解析**：当初「非線形でしか読めない」とされた盤面が、**符号化の座標系を変える（black/white → mine/yours）と線形に読める**ことを発見。→ 教訓：「線形 probe が失敗したら、非線形と結論する前に reparametrization を疑え」。本研究の H2（相対符号化仮説）の直接の祖先。
- **チェス**：AlphaZero / Leela の内部に駒配置・先読みの構造化表現（McGrath et al.; Jenner et al.; Karvonen 2024）。
- **迷路**：Transformers use causal world models in maze-solving (2024–25)。SAE が線形 probe より世界モデルの単離に適するという報告あり。
- **RL 世界モデル**（IRIS/DIAMOND, 2026）：ゲーム状態変数が近似線形に復号可能で、probe 方向への介入が予測を相関的に変化させることを確認。
- **MetaOthello** (2026)：複数ゲームの世界モデルの組織化。ルール重複時に early = ゲーム非依存状態、middle = ゲーム同定、late = ルール特化という層別分業。
- **時系列基盤モデル** (time2time, 2025)：hidden state 介入で稀少事象（暴落）を誘導。

**音楽への適用はゼロ**（2026-07 時点の検索で該当なし）。SynTheory 等の probing（readability 側）は存在するが、「状態への介入 → 下流整合性の検証」という世界モデル判定はなされていない。

## 3. なぜ音楽が世界モデル研究の良い基質か

1. **状態が音楽理論により外的に定義済み**：key は数百年の理論と心理実験（Krumhansl 系）で規定された、モデルから独立した状態変数。Othello の盤面と同じく ground truth が作れる。
2. **合成データで完全な状態ラベル**：機能和声文法から系列を生成すれば、全トークンに正解 key が付く（Othello-GPT が合成棋譜で訓練したのと同型）。
3. **「合法手」の類似物がある**：diatonic scale membership（in-key ratio）は Othello の legal-move rate の自然な対応物。
4. **状態遷移に文法がある（Othello を超える部分）**：転調は任意でなくピボット等の慣習に従う。介入後にモデルが「即座に飛ぶ」か「音楽的な経過を挿入する」かは、Othello には存在しない次元の問いであり、本研究固有の新規性。
5. **座標系の問い（mine/yours の音楽版）**：key の符号化は絶対（C major = 固有方向）か、相対（移調同変：全 key が単一の回転群で結ばれる）か。移調という群作用が明確に定義できる音楽は、この問いに理想的。

## 4. 仮説（すべて反証可能な形で）

- **H1（存在）**：内部活性から key が復号でき、その精度は **入力表層ベースライン（直近窓のピッチクラスヒストグラム / Krumhansl-Schmuckler）を上回る**。特に曖昧領域（転調直後・疎なテクスチャ）で差が開く。
  - 反証条件：どの層でも selectivity 補正後の probe 精度が入力ベースラインを上回らない。
- **H2（符号化形式）**：key 表現は移調同変。すなわち移調 T_k に対し表現空間の線形写像 R_k が存在し、{R_k} が巡回群構造（R_k ≈ R_1^k）をなす。
  - **H2b**：移調 augmentation ありで訓練したモデル（R-Aug）は、なし（R-NoAug）より強い同変性を示す。訓練条件を操作した因果的検証。
  - 探索的：tonic 方向 12 本の幾何は chromatic circle か circle-of-fifths か（Krumhansl 幾何との接続）。
- **H3（因果性・本丸）**：key 部分空間の targeted edit は、継続生成の key を編集先に移す。効果は rank と norm を揃えたランダム部分空間 edit を有意に上回り、かつ musicality guard（参照モデル perplexity）を予算内に保つ。
  - 反証条件：TKR（target-key realization）が統制 edit と区別できない、または品質崩壊としか両立しない。
- **H4（持続と再主張）**：one-shot edit 後の状態は (a) 持続する、(b) 文脈により元の key へ再主張される、(c) ピボット的経過を経て遷移する、のいずれかに分類できる。どれであっても機構的知見（(b) は erasure/self-repair の音楽版）。
- **H5（局在）**：causal に有効な edit 層は限られた層帯に局在する（MetaOthello の層別分業、および自身の Dissociation 計画と接続）。

## 5. 主要な交絡と対策（設計の生命線）

| 交絡 | 何が危ないか | 対策 |
|---|---|---|
| key は表層から推定可能 | probe が「内部状態」でなく「入力の透過」を読むだけかも | C3 入力ベースライン（PC histogram / KS、窓長 sweep）を必ず併記。主張は「ベースライン超過分」に限定。曖昧度で層別 |
| probe 次元の水増し | ISMIR #214 R3 批判と同じ | C1 shuffle selectivity, C2 untrained model, 次元統制 |
| 編集の非特異性 | 「何を押しても音が変わる」 | rank/norm を揃えたランダム部分空間統制、sham edit（null 必須）、12 target の特異性行列 |
| 品質崩壊を「成功」と誤認 | in-key 率は雑音でも上がり得る | 参照モデル perplexity 予算 + 文法統計を常時併記 |
| tokenizer による状態の漏洩 | key/コードトークンが語彙にあると自明化 | 語彙は絶対ピッチ＋時間のみ。key・コード・ローマ数字トークンを**含めない** |
| 合成データの人工性 | 「おもちゃの文法を学んだだけ」 | D-REAL（注釈付き実データ）での probe 検証、M-WILD（公開事前学習モデル）への一般化を Phase C に |

## 6. 投稿戦略との対応

- **ICASSP 2027**（締切 2026-09-16, Toronto）：Phase A + Phase B core（M-CTRL、H1–H3）を 4 ページに。SP 系読者に向け「生成モデルの内部状態の causal 検証」として。
- **OJSP**（regular submission, rolling）：Phase C を加えた完全版（H4, H5, R-Aug/NoAug, M-WILD, metric-position 第二状態）。ICASSP 2027 CFP には OJSP special track（8+1p、同一締切・同一レビュー時程）も存在するが、9 週間で 8p 品質はリスクが高く、**4p ICASSP → 完全版を OJSP regular** を既定路線とする（SPEC §9）。

## 7. ISMIR #214 の教訓の反映（執筆規範）

1. Fig.1 は具体例：楽譜断片、介入箇所の模式図、編集前後の継続の key 推定。抽象語の前に絵。
2. 用語表を Section 2 冒頭に（key state, subspace edit, donor, TKR, IKR, …）。
3. 同一 narrative の反復は 1 回。draft 後に反復検査を必ず回す。
4. take-home 一文：**"Editing a music Transformer's internal key state changes what key it composes in — evidence that it tracks tonality as a causal latent variable, not just surface statistics."**

## 8. 既知の限界（最初から書いておく）

- M-CTRL は小型・合成中心：主張は「この訓練条件の Transformer に tonal world model が創発し得る」であり、商用大規模モデルへの外挿は Phase C まで留保。
- key はスカラーでなくカテゴリ変数：Othello の盤面（64 マス）より低次元で「状態」としては単純。逆に言えば最初の一歩として検証力が高い。
- 実データの key 注釈は一部アルゴリズム由来：D-REAL の結果は secondary と明記。
