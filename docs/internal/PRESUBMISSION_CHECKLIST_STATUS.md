# 投稿前チェックリスト v1 — 対応状況

2026-08-11 (Claude)。対象: ~/ダウンロード/TonalWM_投稿前チェックリスト_v1.md。
数値と凍結済み設計は不変。変更は記述の正確化・引用・post-hoc 補足解析のみ。

## §0 今週やる5件

| # | 項目 | 状況 |
|---|---|---|
| 1 | P_V 直交性 | ✅ **分岐A確定**（下記 §1.1） |
| 2 | syntheory 引用修正+追加引用 | ✅ 完了（§2） |
| 3 | 短調の意思決定 | ✅ **実行・監査・本文反映済み**（2026-08-12, AMENDMENT 3; edit 0.495 vs K1 0.003, 12/12） |
| 4 | 聴取実験の判断+設計凍結 | ⏳ 著者判断待ち（設計テンプレは docs/LISTENING_TEST_DESIGN.md に用意） |
| 5 | 可読性スイープ+spconf 実測 | ✅ スイープ済み / ⏳ spconf.sty は再配布不可のため実機ビルドは著者環境で |

## §1 技術的正確性

- **1.1 直交性 → 分岐A**: v_probe / v_mean は SVD で正規直交化
  （subspaces.py `orthonormal_rows`）、V-DAS は torch 直交拘束。最終テストの
  第4層プローブ V で実測 **max|V^⊤V−I| = 2.4e-7**。§3.2 に
  「We orthonormalize the columns of V …, so P_V is an orthogonal projection.」
  を追加（対訳つき）。
- **1.2 層スイープ**: 各層×**その層自身の**プローブ方向で確認
  （06_sweep.py L173: `v_probe(pw[f"layer_{layer}"])`）。§5.4 の整合文の
  前提どおり。本文変更なし。
- **1.3 検定細部**: リバッタル回答を docs/REBUTTAL_NOTES.md §1–2 に用意
  （zero_method="wilcox" → 不一致対のみの片側符号検定と等価; ガード窓の
  非対称は保守的で意図的）。

## §2 引用（すべてウェブで実物確認のうえ実施）

- **2.1** syntheory2024 の7概念（tempo/拍子/notes/intervals/scales/chords/
  progressions）に key タスク無しを arXiv:2410.00872 で確認 → §2.1 を
  checklist の置換案どおり修正、**castellon2021calm** 追加
  （Castellon & Donahue & Liang, ISMIR 2021 — key detection をプローブ）。
  要旨 S1 は "A small classifier can read…" に変更済みで整合。
- **2.2** **singh2026discovering**（N. Singh, M. Cherep, P. Maes, ICLR 2026,
  arXiv:2505.18186）をステアリング引用群へ追加。**ma2024root**
  （W. Ma & G. Xia, ICML 2024 MI Workshop, OpenReview Kr6nkNa4TQ — MusicGen の
  和音 root/quality に probe+intervene）を追加し差別化1文
  （介入対象は和音表現、調の据え付け+追従ではない）。musicrfm2026 は
  arXiv:2510.19127 で確認: notes/chords への**加算注入** → 既存の §2.1
  差別化（足すだけでは現在の調成分が残る）で被覆、補強不要と判断。
  smitin2024 = SMITIN（attention-head 介入）で attention 系被覆済みを確認。
- **2.3** §2.2 に「音楽ではこの名が加算型ステアリングにも使われる」の1文
  （facchiano2025patching）。facchiano の bib 実在はコンパイルで確認済み。

## §3 主張の較正

- Limitations 冒頭に非主張の1文を追加（checklist の EN 案そのまま）:
  "We claim a causally used internal representation of key, not a human-like
  concept of it: …use, not understanding."
- know/understand を答えとして使っていないことを grep 確認（結論は
  「the music follows the key we write in」で操作的）。タイトルは疑問形で維持。

## §4 短調 — 事前チェック結果（判断材料）

- **標的平均の退化チェック: 通過** — 短調12クラスの寄与位置は 3,365–4,293
  （最大/最小 1.28）、ゼロや極端な偏り無し（mu_balance.json）。
- **コーパスの短音階: 和声的短音階**（generator.py: "major and
  harmonic-minor keys"）→ 音階内割合の定義は和声的短音階で明記可能。
- **KS 相対調混同**: 特異性分析に「相対調」セルを足す準備は既存 parquet で可能。
- **障害**: 現在この計算機の GPU がドライバ不整合（NVML mismatch）で使用不可。
  再起動またはドライバ修復が先。規模感は長調の held-out ラン
  （5アーム×1,200生成）と同等。
- **実行は著者の意思決定**。走らせる場合は FREEZE 文書に AMENDMENT を追記
  （事前登録済み二次条件の実行なので探索やり直しではない）→ スクリプトの
  MAJOR_TARGETS を短調12クラスへ拡張する変更が必要。

## §5 標準性の係留（本文1句ずつ、実施済み）

- §4.1: SynTheory の合成・概念単離設計に倣う旨の1句。
- §4.3: KS =「標準のプロファイル照合法」1句。
- §4.2: 公開モデル選定理由（公開 CP / 記号領域 / 標準コーパス / 標準評価対）1文。
- 相場観対応表（MusicRFM の評価設計との対応）→ REBUTTAL_NOTES.md §4。

## §6 聴取実験

- 設計テンプレを **docs/LISTENING_TEST_DESIGN.md** に用意（Singh et al. を
  土台に、#214 の検定力不足を修復: ≥20名×≥10セット、事前の検定力計算、
  音楽経験の報告）。**凍結はまだ**——実施判断とともに著者がコミットする。
- 実施不能時の防御は実装済み: Limitations に客観判定可能性の1句、
  音源デモ公開は匿名リポジトリ作成時の TODO として記録。

## §7 可読性

- 要旨1語目 "Probes" → "A small classifier"（定義前使用の解消）。
- steering の §1 初出は「§2 参照」つきで許容と判断（記録）。
- 手続き説明の重複スキャン: 探索/最終テストの手順説明は §4.4 のみ、
  他は参照のみ（grep 確認）。
- 冒頭2ページの非専門家テスト: **著者側の工程**（実施推奨のまま残す）。

## §8 体裁・再現性

- 図はベクター（matplotlib PDF, fonttype 42）✅。(a)(b) ラベルは3図とも
  図中に明記 ✅。
- 0.71 → **0.710** 桁揃え、§5.1 ダッシュ閉じ（括弧化）、音高のみ探索段階に
  床 **0.075** を明記 ✅。
- 匿名リポジトリ+URL 差替え+Paper ID: ⏳ 著者作業。
- spconf 実機ビルド: ⏳ spconf.sty 非再配布のため著者環境で。フォールバック
  実測: 要旨 66.1mm（<80mm）、6ページ。
- 文献最終確認の★リマインダー: tex に維持。

## §9 既存データだけの追加解析

- **9a 五度圏距離分解: 実施** → results/confirmatory/R-Aug_s0/
  fifths_distance_posthoc.json + 台帳（post-hoc 明記）。編集は距離に
  フラット（0.295–0.425）、K1 は距離1に集中（0.200、他 ≤0.01）=
  「近い調へ押しているだけ」への強い反証。REBUTTAL_NOTES.md §3 に表。
- **9b 層別伝播チェック: 未実施**（GPU 不可のため）。設計: 第4層編集の
  teacher-forced プレフィクスで第5–7層の活性を capture し、各層プローブの
  予測が標的調へ変わる割合を測る。フォワードのみ・探索プロンプト・補足行き。
  GPU 復旧後に実装可（新スクリプト 30 番台）。

## §10 維持すべき強み

- 読める≠使える の骨格・低次元線形介入・誠実性の開示はすべて無傷。
  §1.1 が分岐Aで片付いたので「24次元の直交射影による低次元介入」は
  むしろ強みとして主張できる。
