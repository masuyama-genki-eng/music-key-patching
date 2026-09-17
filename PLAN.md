# PLAN — ICASSP 2027 最終改稿（ブランチ `icassp27-revision`）

作成 2026-09-17。締切 2026-09-23。Phase 0 の成果物。**ここで一度止まり、§6 の確認事項に回答をもらってから Phase 1 に入る。**

前回の追加実験計画は `docs/PLAN_ADDITIONAL_EXPERIMENTS_2026-09-10.md` に退避した（内容は不変）。

---

## 0. 環境（指示書の [要記入] を実物で埋めた）

| 指示書の記号 | 実体 |
|---|---|
| [REPO_PATH] | `/home/masuyama-genki/ICASSP③/tonal-world-model` |
| [TEX_DIR] | `paper/`（`icassp2027.tex` / `icassp2027_supp.tex` / `spconf.sty` / `refs.bib`。`main.tex` という名前のファイルは無い） |
| [CKPT_DIR] | `results/models/{R-Aug,R-NoAug}_s{0,1,2}/final.pt`（6本） |
| [MAIN_CKPT] | `results/models/R-Aug_s0`（probe/μ は `results/probing/R-Aug_s0/`） |
| [REF_CKPT] | `results/models/M-REF_s100`（閾値 0.613 は `results/guard/delta_ppl.json`） |
| [PROBE_DATA] | `results/data_syn/test.parquet` 行 0–5999（probe と μ）。探索プロンプトは行 0–167 |
| [FINAL_PROMPTS] | 同 parquet 行 ≥ 6000 の stable-major 先頭100（行 6000–6177）/ stable-minor 先頭100（行 6004–6417）。`confirmatory_test.select_prompts_holdout` が凍結規則 |
| [CALIB_DATA] | `results/data_syn/val.parquet`（自然転調イベント、`guard/delta_ppl.json` の元） |
| [PUBLIC_CKPTS] | AMT `stanford-crfm/music-{small,medium,large}-800k`（=12L/24L/36L）、MMT `data/mmt-checkpoints/mmt/lmd/ape`、REMI+ `remi-lmd-remi` |
| [BACH_DATA] | `data/bach-370-chorales` + `data/When-in-Rome`（300曲）。220/80 分割は `results/mwild_bach_pooled80/<ckpt>/balanced/balanced_report.json` の `split_info` に固定 |
| [POP_DATA] | `configs/pop909.yaml`（905曲、search/final 分割） |
| [PUBLIC_REF] | Bach: `music-medium-800k`（0.8489 nat）、Pop: `music-large-800k`（1.1185 nat） |
| [RESULTS_DIR] | `results/`（台帳 `RESULTS_LEDGER.md`、追記専用） |
| [GPU_SPEC] | RTX 6000 Ada 48 GB ×1。現在アイドル |
| [N] | 未測定。自前 25M モデルは 1 ジョブ ~1.5 GB なので **3 並行を提案**（Phase 1 冒頭に 10 分の並行スループット測定を入れ、実測で決める） |

---

## 1. プロトコル照合（指示書 §「変更禁止」× コード）

| 項目 | 指示書 | コード / artifact | 判定 |
|---|---|---|---|
| モデル | 8層 GPT-2 型, 512次元, ~25M, 調・和音トークン無し | `src/model/gpt.py`, `src/tokenizer/vocab.py` | ✅ |
| プローブ | 24クラス多項LR, V = W の行空間 SVD rank 24 | `subspaces.v_probe(W, rank=24)` = `orthonormal_rows` | ✅ |
| 介入 | h' = h − Ph + Pμ, μ は目標調ラベル位置の平均（種別不問）, 9バー目以降全位置 | `SubspaceEditor(mode="replace")`, `mu_targets_from_means` | ✅ |
| 対照（主継続） | 次元一致 + 変位一致 | `k1_norm`（`norm_ref=V`）。本文 Fig.2 左の 5.6/2.6 は k1_norm ✅ |
| 対照（層走査・次ピッチ） | 次元一致のみ | 層走査 CSV `edit_control = K1 (NOT displacement-matched)`, next_pitch は `k1` | ✅ |
| **対照（公開モデル）** | 次元一致のみ | Table 2 の 60 プロンプトセルは `k1` ✅。**Fig.4 左（Bach pooled80）は `k1_norm`** | ❌ **食い違い D1** |
| **公開モデル SR の尤度基準** | 平均NLL差 ≤ 0.85 (Bach) / 1.12 (Pop) | Table 2 セル: guarded ✅（AMT-12L 0.4792, guard_pass 0.997）。**Fig.4 左の SR は stage-1 走査の生の一致率で、その行に NLL 差は計算されていない**（`stage1_layer_rows.json` に `nll_excess` 列が無い。`fig_public_layerwise_corrected.py:170` も `edit_guarded=False` と明示） | ❌ **食い違い D1** |
| 層選択 | 探索プロンプトで実施、最終プロンプトは不使用 | 自前: 行 0–167 で走査 ✅。**Fig.4: 層を選んだ 80 プロンプトと報告する 80 プロンプトが同一**（`stage1_layer_scan.json` の note が「not a disjoint final-test evaluation」と自己申告） | ⚠️ **D3** |
| 最終テスト | 100+100 × 11 = 1,100/モード, T=1.0, top-p 0.95, 384 tok, 1 sample, KS は全体 (EOS/16バー) | `configs/gen.yaml`, `pitches_and_bars(max_bars=16)`, `GEN_SEED=7` | ✅ |
| SR | KS == 目標 ∧ E = exp H(y') − exp H(y) ≤ 0.613 | `guard.guarded_success`, `mref_ppl_excess` | ✅ |
| 次ピッチ | teacher forcing で bar/pos 付加 → 介入 → D, δD | `next_pitch_test.py` | ✅ |
| 検定 | 目標調ごと片側 Wilcoxon, Holm 12 | `stats.wilcoxon_rank_biserial`, `holm_correct` | ✅ |
| 公開継続セル | 探索20 / 保持60 / 長調12 | Table 2 ✅（`--n-prompts-stage1 20 --n-prompts-stage2 60`） | ✅ |
| Bach 220/80 | | `balanced_report.split_info`（80 = pooled prompt 名を除外した 220 で推定） | ✅ |
| T0 の再現目標値 | 35.5/49.5, 5.6/2.6, +0.695/+0.309, +0.041/+0.026 | `verdict.json`: 0.3555/0.4945, k1_norm 0.0564/0.0264; `next_pitch.json`: 0.6953/0.3094, 0.0410/0.0261 | ✅ 数値は一致 |

### 止まって報告する食い違い

- **D1（重大）** 本文 3.3「公開モデルは平均NLL差で閾値」「公開モデル評価は次元一致対照」と、Fig.4 左の実体（**ガード無し・K1-norm**）が一致しない。継続トークンが保存されていないので事後にガードを掛けることはできない。選択肢は (a) Fig.4 のピーク層（AMT L8 / MMT L5 / REMI+ L5）を 80 プロンプトでガード付き・k1 対照で**再生成**（見積 ≈ 1.5 h、§4）、(b) Fig.4 キャプションと 3.3 を「層走査は生の一致率、変位一致対照」と書き換える。**どちらにするか指示が必要。**
- **D2** `fig_main_results_bars.py` は SR を **ハードコード**（`sr_repl = [35.5, 49.5]`, `sr_ctrl = [5.6, 2.6]`）。値は正しいが T7 の出所表と矛盾するので、`verdict.json` を読む形に直す（T6 の図再生成で対応、確認不要）。
- **D3** Fig.4 左は「層を選んだプロンプトで報告」。本文 3.3 は 80 曲を評価に使ったと述べていて虚偽ではないが、選択と評価が同一集合である旨は書かれていない。D1 で (a) を選ぶなら同じ再生成で解消はしない（層は既に選ばれている）ので、1 句の追記が必要。
- **D4** 継続トークン列は最終テストでは保存されていない（parquet は指標のみ）。指示書「生成した継続は全て保存」に従い、新規実行分は保存する（既存 artifact は不変）。
- **D5** `next_pitch_test.py` に `--tag`/`--outdir` が無く、T0 の再実行が台帳登録済み `results/confirmatory/R-Aug_s0/next_pitch.json` を**上書き**する。T0 前に追加する（既定動作は不変）。
- **D6** 作業ツリーに著者の未コミット変更（tex 617 行、公開モデル系スクリプト 4 本、新図スクリプト 8 本）がある。ブランチは切ったが、このままだと私の最初のタスクコミットに混ざる。**Phase 1 の最初に「baseline: author WIP 2026-09-17」として 1 コミットにまとめてよいか確認したい。**
- **D7** 現状の本文は **4 ページ + 参考文献 1 ページに既に収まっている**（latexmk 5 ページ、4 ページ目末尾が Conclusion、5 ページ目は REFERENCES のみ）。T6 は「6–8 行足して同量以上削る」の運用になる。
- **D8** seed 1 の held-out 再現は 2026-09-10 に完了済み（`results/confirmatory/R-Aug_s1/verdictf3.json`: L2, edit 0.2891 vs K1 0.0164, 12/12, BCa [0.2418, 0.3027], guard 0.735）。ただし K2 sham ゲートは **AMENDMENT 3 の許容（logit 差 < 1e-4 を 100 件中 1 件）** で通過している。T1 で再利用するか再走するか（§6）。

---

## 2. タスク → 既存実装 対応表

| タスク | 既存スクリプト / 関数 | 不足 | GPU 見積（逐次） |
|---|---|---|---|
| **T0 再現** | `experiments/confirmatory/confirmatory_test.py --arms edit --tag t0`（k1, k1_norm は常に同走）、`--mode minor`；`next_pitch_test.py --mode {major,minor}` | `next_pitch_test.py` に `--tag`；継続保存（D4）；**clean 行のスコアと保存**（T2 と共用、生成は既にゲートで行っている） | 長調 38 分 + 短調 38 分 + 次ピッチ 2 分 ≈ **1.3 h**（2 並行で ≈ 40 分） |
| **T2 天井** | T0 の同一実行で `clean` 条件を parquet に残す（`SW.rows_for_condition({"cond":"clean"})` は既に呼ばれているが行を捨てている） | 新 `experiments/reanalysis/ceiling.py`（parquet → (a) 元調一致率, (b) 他11調平均 = 偶然水準, モード別）→ `results/ceiling.{json,md}` | **0**（T0 に同乗） |
| **T4 rank-23** | 基底: `probe_centering_audit.py` の `Wc = W − mean(W)` の行空間 = 指示書の v = Wᵀ(WWᵀ)⁻¹1 の直交補（同値、証明を PLAN §3 に記す）。編集: `confirmatory_test.py`, `next_pitch_test.py` | `subspaces.v_probe_centered(W)`（新関数）と両スクリプトに `--basis {rank24,rank23}`；K1 は rank 23 で自動一致（`k1_basis(V)` が V の階数を見る） | 長調 38 + 短調 38 + 次ピッチ 2 ≈ **1.3 h** |
| **T1 6走行** | probe/μ: 6 モデルとも `results/probing/<m>/{probe_weights,class_means}.npz` **済**。層走査: `experiments/editing/edit_sweep.py --model-dir … --methods v_probe,k1_r24`（R-Aug_s0, R-Aug_s1 は済、選択層 4 / 2）。最終テスト: `confirmatory_test.py --model-dir --probing-dir --layer`。次ピッチ: `next_pitch_test.py --model-dir --probing-dir --layer` | 集約 `experiments/reanalysis/seed_robustness.py` → `results/seed_robustness.{json,md}`；R-Aug_s1 は K2 ゲート許容が必要（`--gate-tolerance`、s0 と同じ規則で他 seed も落ちる可能性あり → 落ちたら止めて報告） | 層走査 4 モデル × 8 層 × (v_probe + k1) × 12 分 = **12.8 h**（v_probe のみなら 6.4 h）；最終テスト 5 モデル × 2 モード × 38 分 = **6.3 h**（s1 長調を再利用なら 5.7 h）；次ピッチ 10 分。**合計 ≈ 19 h 逐次、3 並行で ≈ 7–8 h** |
| **T3 公開 PC 対照** | 活性抽出 `src/probing/public_model.extract_activations`（`pitch_windows` 付き）；PC 部分空間 `pitchclass_subspace.py` の `ridge/hist_features/orthonormal`（w=16, 512 の 12 行ずつ → 24 行スタック、λ は探索側で選ぶ）；編集 `public_edit_sweep.py --stage 2 --layers L --artifacts-dir <pc_dir> --control k1` は `probe_weights.npz` の行空間を V に取るので、**24 行の PC 基底を `layer_L` として書いた npz を渡せば無改造で流れる**。μ は元の `class_means.npz` をコピー | 新 `experiments/public_models/public_pitchclass_subspace.py`（fit + npz 出力 + 残差基底変種）；集約 → `results/public_pc_control.{json,md}` | 層固定 1 層 × 60 プロンプト × 12 調 × 2 条件: AMT ≈ 40 分, MMT ≈ 15, REMI ≈ 15 → **≈ 1.2 h**；残差変種 +1.2 h；Pop 3 モデル +≈1.5 h |
| **T5 引用** | `paper/refs.bib`（`leace2023` は既にある） | INLP / Wang et al. / Zhang & Nanda を追加、式(3) 直後に 1 文 | 0 |
| **T6 原稿** | 図: `fig_main_results_bars.py`, `fig_layerwise_gpt2_stacked.py`, `fig_public_results_bars.py`, `fig_public_layerwise_*`（CSV は `results/layerwise_*.csv`） | 2×2 統合図スクリプト or Fig.4 → Table 2 吸収；本文編集 | 0 |
| **T7 チェック** | `collect_paper_numbers.py --check-tex`（現状 **0 未追跡**、両文書）；`latexmk`；`pdftotext` | `results/PROVENANCE.md`、`paper_edits.md` | 0 |

---

## 3. 不足実装の仕様（すべて既定動作不変・追加のみ）

1. `next_pitch_test.py --tag` … 出力を `next_pitch{tag}.json` / `next_pitch_L{L}{tag}.parquet` に。
2. `confirmatory_test.py --save-conts` … 各条件・目標・プロンプトのトークン列を `parts/conts_L{L}{tag}.json.gz` に保存。`--keep-clean-rows` … ゲート用に生成済みの無介入継続を `cond="clean"` として同じ parquet に残す（SR 計算からは `identity` と同様に除外）。
3. `subspaces.v_probe_centered(W)` … `Wc = W − W.mean(0)` の行空間 SVD、rank 23。v = Wᵀ(WWᵀ)⁻¹1 との関係: Wc の行空間 = {Wᵀc : cᵀ1 = 0} で、⟨Wᵀc, Wᵀ(WWᵀ)⁻¹1⟩ = cᵀ1 = 0 なので、これは W の行空間 ∩ v⊥ に一致する（次元 23）。単体テストで `V23ᵀ v ≈ 0` と `P23 ⊂ P24` を固定する。監査済み artifact `results/reanalysis/probe_centering_audit/` の「extra/P23 mean 0.088–0.238」が本文の 9–24% の出所。
4. `edit_sweep.py` はそのまま使用（`--methods v_probe,k1_r24`）。選択規則は主モデルと同一: **v_probe の guarded SR が最大の層**（L4 は 0.378 でこの規則で選ばれた）。
5. `public_pitchclass_subspace.py` … 220 曲（`balanced_report.split_info` の推定側）で `extract_activations(probe_at="predict_pitch")` → `pitch_windows` から w=16/512 の L1 正規化ヒストグラム → ridge（切片あり、λ は AMT の探索 20 プロンプト位置で選び 3 モデルで固定）→ 12 行 × 2 窓 = 24 行 → `layer_L` に書いた `probe_weights.npz` と元 `class_means.npz` のコピーを `results/public_pc_control/<ckpt>/` に置く。残差変種は V から PC 部分空間を射影除去した基底（rank は落ちた分だけ）。
6. `seed_robustness.py`, `ceiling.py`, `public_pc_control` 集約 … いずれも `results/` の JSON/parquet だけを読む。

---

## 4. 実行順と壁時計見積

```
Phase 1a  並行スループット測定（10 分）→ N を確定
Phase 1b  T0 長調 ‖ T0 短調 ‖ T4 長調        ≈ 40 分
          T4 短調 ‖ T1 層走査 R-Aug_s2 ‖ R-NoAug_s0
Phase 1c  T1 層走査 R-NoAug_s1 ‖ R-NoAug_s2 ‖ T1 最終 R-Aug_s1 短調
Phase 1d  T1 最終テスト ×(4 モデル × 2 モード) 3 並行
Phase 1e  T3 fit（CPU/GPU 数分）→ AMT ‖ MMT ‖ REMI+
```
逐次合計 ≈ 23 h。3 並行で **≈ 9–10 h**（GPU が飽和しなければ）。D1(a) を選ぶ場合 +1.5 h。

30 分超のジョブは **T0, T4, T1 の各層走査・各最終テスト, T3 の AMT** で、いずれも上表の見積を提示済み。§6 の回答をもって投入確認とみなしてよいか、ジョブごとに個別確認が要るかも §6 で聞く。

---

## 5. T6 で触る箇所の地図（Phase 2 用メモ、今は実行しない）

- 4.2（`sec:h3`）: 6 走行の範囲 1 文、天井 1 文、rank-23 半文。
- 4.5（`sec:public`）: SR_pc 1 文。Table 2（`tab:public-extra`）に SR_pc 列。
- Fig.2 と Fig.4 を 2×2 に統合 or Fig.4 → Table 2 に δD 列として吸収。Fig.2 のハードコードは撤去（D2）。
- Limitations: 「highest success rate among the three augmented seeds」（T1 で範囲を書けるので削除候補）、「9–24% … have not tested」（T4 で結果に変える）。
- Sec.1 第 2 段落を 3–4 行圧縮、3.3 Evaluation prompts の数値を統合図キャプションへ。
- Abstract 末尾 / Conclusion の提案文 2–3 案は **著者選択**。
- 文字サイズ: Table 2 が `\scriptsize` で 9pt 未満 → T7 規則に抵触。`\tabfont`（9/10.5）に替える必要あり（要確認: 既存の著者判断か）。

---

## 6. 確認事項（回答待ち）

1. **D1**: Fig.4 左を (a) ガード付き・k1 対照でピーク層のみ再生成（≈1.5 h）するか、(b) 本文とキャプションを実体に合わせて書き換えるか。
2. **D6**: 著者の未コミット変更を `baseline: author WIP 2026-09-17` として最初にコミットしてよいか（内容は一切変えない）。
3. **T1 の層走査**: 主モデルと同じ `v_probe + k1_r24` の 8 層（12.8 h）か、選択に必要な `v_probe` のみ（6.4 h）か。K1 は選択には使わないが、モデル別の層別 edit margin を `.md` に載せるなら必要。
4. **D8**: R-Aug_s1 長調は 09-10 の結果（許容付きゲート）を再利用してよいか。他 seed でゲートが 1e-4 許容でも落ちた場合は止めて報告する。
5. **T3 の範囲**: 必須 = Bach 3 モデル PC24（1.2 h）。残差変種（+1.2 h）と Pop 3 モデル（+1.5 h）は「時間があれば」の扱いで、Phase 1d の後に着手可否を改めて聞く形でよいか。
6. **30 分超ジョブの確認方式**: 本 PLAN の見積提示で一括確認とするか、投入ごとに確認するか。
7. Table 2 の `\scriptsize` を `\tabfont`（9pt）に上げてよいか（行が増えるので T6 の削減量に影響）。

---

## 7. 承認と確定事項（2026-09-17、Phase 1 開始前に記録）

PLAN c87c6b9 は承認された。§6 への回答:

1. **D1/D3 → (a) 再生成**、次の規則で D3 も同時に解消する。**この規則は集計を行う前にここに書いてコミットする。**
   - pooled80 の 80 曲（`stage1_layer_scan.json` の `prompt_names`、3 モデルで同一・chorale ID 昇順）を、**ID 昇順の先頭 20 曲 = search、残り 60 曲 = final** に分ける。search = chor001, chor005, chor006, chor007, chor010, chor013, chor016, chor017, chor020, chor022, chor024, chor026, chor027, chor028, chor029, chor030, chor031, chor032, chor033, chor036。
   - 既存 stage-1 走査の per-prompt 行（`stage1_layer_rows.json`、80 曲 × 全層 × 12 調 × {edit, k1_norm}）を **search 20 曲だけ**で再集計し、各モデルの edit margin（edit − K1-norm）ピーク層を選ぶ。走査の対照が K1-norm しか無いことは承知の上で選択にはそれを使う。
   - 選んだ層で **final 60 曲**を再生成: replacement, K1（次元一致、本文 3.3 に合わせる）, K1-norm。NLL ガードは凍結の参照割当と閾値（AMT-12L / MMT / REMI+ → `music-medium-800k`, 0.8489）。**継続トークンを保存する。**
   - Fig. 4 右の δD も同じ 60 曲・同じ層・K1 対照で再計算する。
   - 20 曲でのピークが 8/5/5 と違えば **20 曲の結果に従う**。80 曲走査は補足に「exploratory layer profile（raw key-hit, K1-norm）」として残す。
   - 本文 3.3「Evaluation prompts」と Fig. 4 キャプションを 220/20/60 の実体に書き換える案を T6 で出す。4.5 節の数値と δD は再生成値に置換。
2. **D6** → `baseline: author WIP 2026-09-17`（44d6147）として単独コミット済み。以後 WIP は混ぜない。
3. **T1 層走査** → `v_probe + k1_r24`、8 層（12.8 h）。選択基準は本文と同じ **edit margin**（探索プロンプトのみ）。補足 §23 の層 4 固定 4 走行は「transfer 条件」として残し、再選択結果と両方報告する。
4. **R-Aug_s1 長調** → 09-10 の replacement と K1 を再利用。欠けるアーム（K1-norm、短調）のみ追加タグで実行。AMENDMENT 3 の許容通過は `results/seed_robustness.md` に明記。
5. **T3 残差変種・Pop** → 再確認不要。GPU が空いた時のみの低優先ジョブ。**9/20 終業までに完了・検査通過したものだけ**論文に入れる。
6. **30 分超ジョブ** → PLAN 記載分は一括承認。止まる条件は「実測が見積の 1.5 倍超」「失敗/OOM」「PLAN にないジョブ」。進捗は各ジョブ完了時に 1 行。
7. **Table 2** → `\small`（9pt）。収まらなければ、列名略記 → AMT-24L/36L 行を本文 1 文＋補足へ、の順。

追加指示: D2 の `verdict.json` 読み替え、D4/D5 の追加専用オプションは了解。**以後の全ジョブで継続トークンを保存する。**実行順は D1 再生成 ‖ T0(+T2) ‖ T4 → T1 走査（夜間）→ T1 最終テスト → T3 必須分。T0 が 4 組の目標値を再現するまで T1 の数値は暫定。補足に「Deduplicated Bach evaluation (220/20/60)」節の下書きを T6 で追加し、§1・§7・§26・§32 の「final prompts は target estimation に含まれる」記述を Table 1 セル限定に修正する。

### 実測メモ（2026-09-17 23:17）
- 自前モデルの最終テストを 3 並行で投入したところ GPU 使用量が 46.9 GB に達し、3 ジョブとも K2 ゲート中に CUDA allocator の OOM 警告（回復はした）。1 ジョブの実メモリは 7–21 GB（キャッシュ込み）で、§0 の 1.5 GB という見積は誤り。**T4 長調を停止**（成果物・台帳への書き込みは無し）し、**確認テストは 2 並行（[N]=2）** とする。次ピッチ（0.7 GB）は追加で同走可。T4 は T0 完了後に再投入。
