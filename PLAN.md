# PLAN.md — 追加実験 A〜E の実装計画

指示書 §0.1 の要求（機能の対応表、未実装項目の明示）と §0.2（ハイパーパラメータの
決定過程）に対する記録。作成 2026-09-09、リポジトリ HEAD `3e53bc7`。
再現確認は REPRO.md（6項目すべて ±0.001 以内で合格）。

placeholder が未置換だったため次のとおり解釈した:
`<REPO_ROOT>` = このリポジトリをチェックアウトしたディレクトリ、
`<GPU_SPEC>` = 実機の NVIDIA RTX 6000 Ada 48GB（driver 580.173.02）。

---

## 1. 機能の対応表（§0.1 (a)–(i)）

### (a) 合成データ生成器と各トークンの真の調ラベル
| 機能 | 場所 |
|---|---|
| 1曲の生成（tokens, key_labels, events を返す） | `src/datagen/generator.py:128` `generate_piece` |
| 調ラベルの値域 | `generator.py:59` `Key.index24`（0..23 = 12主音 × 長短）、全トークンに付与（`:233` で長さ一致を assert） |
| 分割の生成 | `experiments/data_and_models/generate_corpus.py:135` `main`、決定性は `SPLIT_OFFSET`（`:47`） |
| parquet スキーマ | `generate_corpus.py:50` — `token_ids`, **`key_labels`**, `initial_key`, `n_key_changes`, `mod_types`, `mod_fifths` ほか |
| 設定 | `configs/data_syn.yaml`（200k/10k/10k + 参照用、base_seed 20260711） |
| 実曲側の同名フィールド | `src/datagen/dreal.py:253` `local_key_labels` |

### (b) モデル読込
| 対象 | 場所 |
|---|---|
| 自前6モデル（3seed × aug/noaug） | `src/probing/extract.py:38` `load_model`、アーキテクチャ `src/model/gpt.py:30` `TonalGPT`、命名は `experiments/data_and_models/train_models.py:57` → `results/models/R-{Aug,NoAug}_s{0,1,2}/final.pt` |
| 第7モデル（擾乱評価の参照） | `train_models.py:60`（`ref_train/ref_val` 分割、seed 100）→ `results/models/M-REF_s100/final.pt`、使用は `src/intervene/sweep.py:132` `mref_ppls` |
| AMT 12/24/36L | `src/publicmodels/anticipatory.py:237` `AnticipatoryAdapter.load`（HF `stanford-crfm/music-{small,medium,large}-800k`） |
| MMT | `src/publicmodels/mmt.py:87` `MMTAdapter.load` |
| REMI | `src/publicmodels/remi.py:76` `RemiAdapter.load` |
| アダプタ契約 | `src/publicmodels/base.py:24` `PublicModelAdapter`（必須14メソッド: load / check_vocab / n_layers / d_model / context_length / vocab_size / block / residual_streams / encode_events / probe_offset / decode_pitches / decode_events / _generate_step / encoding_is_sane）。登録は `registry.py:9`。既存アダプタは3つのみ |

### (c) 活性の取得・置換
| 機能 | 場所 |
|---|---|
| 読み（自前） | `src/model/gpt.py:48` `forward(..., capture=True)` が各ブロック出力 = h_ℓ(t) を返す。収集は `src/probing/extract.py:94` `extract` |
| 読み（公開） | `base.py:75` `residual_streams`、駆動は `src/probing/public_model.py:25` |
| 書き（自前） | `src/intervene/edit.py:28` `SubspaceEditor`（`mode ∈ {replace, sham, add_matched, add_fixed, add_contrast}`、`from_position/until_position/token_mask/norm_ref/mu_source/alpha/s_bar`）。工場関数 `sweep.py:154` `make_editor` |
| 書き（公開） | `src/intervene/public_model_edit.py:29` `HookSubspaceEditor`（`V, mu_target, mode, from_position` のみ） |
| 生成の入口 | 自前: `gpt.py:63` `generate(..., editors=, edit_from=)` / `sweep.py:71` `generate_batch` / `token_masks.py:38` `generate_masked`。公開: `base.py:154` `generate(...)`（フックを `block(model, layer)` に登録） |
| トークン種別限定の編集 | `src/intervene/token_masks.py:28` `MASKS = {all, pos_pitch, pitch, bar_dur}` — **自前モデルのみ**。公開モデルは **未実装**（下記 §3） |

### (d) 調部分空間 V、P_V、µ_κ*
| 機能 | 場所 |
|---|---|
| probe 重み版 | `src/intervene/subspaces.py:35` `v_probe` |
| per-key 平均版 | `subspaces.py:38` `v_mean` |
| 編集最適化版（DAS） | `subspaces.py:64` `train_das` / `:50` `DASSubspace`（QR で直交拘束） |
| 正規直交化 | `subspaces.py:28` `orthonormal_rows`（SVD）＋ `edit.py:22` `orthonormalize`（QR、エディタ内でも再適用） |
| P_V の適用 | 行列は作らず `(x @ V) @ V.T`（`edit.py:118` `_proj`） |
| µ_κ の出所 | `subspaces.py:43` `mu_targets_from_means` = 生のクラス平均（射影はエディタ側）。計算は `experiments/probing/probe_key.py:221` |
| 保存場所 | `results/probing/<model>/class_means.npz`（キー `layer_{ℓ}`）、`probe_weights.npz` |

### (e) KS 推定器・成功判定・擾乱・τ
| 機能 | 場所 |
|---|---|
| KS 推定器 | `src/eval/keyest.py:22` `ks_scores`（KK1982 プロファイル）、`:35` `estimate_key`（argmax、0..23） |
| 成功判定 | `src/eval/guard.py:27` `guarded_success(key_hit, ppl_excess, delta_ppl)`。key_hit は完全一致（`sweep.py:126` `tkr_strict`）。推定不能（音高8未満）と NaN は失敗 |
| 擾乱（nats） | `guard.py:70` `token_nlls` → `:118` `continuation_ppl`、行ごとの `mref_ppl_excess = ppl_edit − ppl_clean_twin`（`sweep.py:120`） |
| τ = 0.613 の算出 | `experiments/editing/freeze_quality_guard.py:55` — `modulation_ppl_rises`（`guard.py:95`、境界の前後 W=24 トークン）の 90 パーセンタイル。上書きは `:45` で拒否 |
| τ の保存 | `results/guard/delta_ppl.json`（+ 公開モデルは `results/mwild_sweep/<ckpt>/delta_ppl.json`） |

### (f) in-key note share
`src/eval/keyest.py:53` `in_key_ratio` / `metrics.py:73` `ikr_pair`。音階集合は
`keyest.py:49` — 長調 `{0,2,4,5,7,9,11}`（7音）、短調 `DIATONIC_MINOR_UNION
{0,2,3,5,7,8,9,10,11}`（9音 = 自然・和声・旋律的短音階の合併）。**3形態を個別に
選ぶ機能は無い**（補足 §18 の和声的短音階のみの再計算はその場で音階集合を作っている）。

### (g) 統計
| 機能 | 場所 |
|---|---|
| 対応あり片側 Wilcoxon + rank-biserial r | `src/analysis/stats.py:42` `wilcoxon_rank_biserial`（`alternative="greater"`） |
| Holm 補正 | `stats.py:58` `holm_correct` |
| prompt/piece レベル BCa bootstrap | `stats.py:11` `bca_ci(units, stat_fn, n_boot=10000, seed=0)` |

### (h) プロンプトと seed
| 対象 | 場所・範囲 |
|---|---|
| 探索 100 prompts（自前） | `src/intervene/sweep.py:32` `select_prompts` → test rows 0–167、seed 既定 0 |
| 最終テスト 100 prompts（自前） | `experiments/confirmatory/confirmatory_test.py:51` `select_prompts_holdout`、`HOLDOUT_START=6000`、`GEN_SEED=7`。実測 長調 rows 6000–6177、短調 6004–6417 |
| 公開モデル | `experiments/public_models/public_edit_sweep.py:52` `build_prompts`、探索20/取り置き60、seed 0（`rng.manual_seed(seed*1_000_003 + pi)`） |
| ランダム対照基底 | `sweep.py:161` `k1_basis(V, seed + 31*layer)` |

### (i) 既存の per-continuation ログ
| 条件 | 場所 | 擾乱値 | 再採点可否 |
|---|---|---|---|
| 長調 最終テスト 5アーム（edit/pitch/bar_dur/k1/k1_norm） | `results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet`（6,000行） | あり | **可** |
| 短調 最終テスト 5アーム | `results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4.parquet`（6,000行） | あり | **可** |
| 移調参照（K4）長調・短調 | `.../parts/k4_ceiling.parquet`（各1,200行） | **無し**（K4 に guard は未定義） | 調のみ可、閾値再採点は不可 |
| steering B/C/D 最終テスト | `results/steering/R-Aug_s0/parts{,_minor}/final_{B,C,D}.parquet`（各1,200行） | あり | **可** |
| steering の α グリッド {0.25,0.5,1,2,4,8,16} × 8層 | `results/steering/R-Aug_s0/parts/C_L{ℓ}_a{α}.parquet` ほか — **探索プロンプトのみ** | あり | **可** |
| 公開5モデル ×（Bach, pop）edit/k1 | `results/mwild_sweep{,_pop909}/<ckpt>/stage2_eval{,_balanced}.json` の `"rows"`（各1,440行、`nll_excess`, `est_key`） | あり | **可** |
| 継続トークン列そのもの | `results/rescore/continuations_R-Aug_s0{,_minor}_L4.json`（edit/k1_norm/clean/reference のみ。pitch/bar_dur/k1 は無い） | — | 推定器の差し替え再採点が可能 |

---

## 2. 【重要】既存成果との重複（無駄な再計算を避けるための棚卸し）

指示書の A〜E のうち、**B・D の大半と E の一部は既に実行済み**だった。以下は
「既存で答えが出ている部分」「本当に欠けている部分」の切り分け。

| 指示書の項目 | 状況 | 根拠 |
|---|---|---|
| **B**（閾値感度） | **主要部分は済** — 長調・短調の edit と k1_norm について、閾値グリッド 0.2–1.6（0.1 刻み）+ 凍結 τ で SR を再計算した 32 レコード/mode が既にある | `results/reanalysis/a1_a3/fifths_L4.json` → `modes.*.threshold_sensitivity` |
| B の欠け | ①他アーム（pitch/bar_dur/k1）②steering B/C/D ③公開モデル ④τ をパーセンタイル（p80/p85/p95）で名付けること | ①②③は per-row ログがあるので**再解析のみ**。④は per-event 分布が未保存（下記 §3-1） |
| **D1**（mode 別 in-key share） | **済** | 補足 §18、`results/reanalysis/a5/minor_handling.json`（union 0.957/0.764、和声的短音階のみ 0.903/0.602、長調 0.936/0.602） |
| **D2**（KS 混同・平行長調） | **概ね済** | 補足 §11 の着地表（長調 relative 0.089）、短調の相対長調セル 4.1%（parquet 導出）、移調参照の mode 別 accuracy = K4 天井 0.648/0.877 |
| **D3**（代替推定器） | **長調は済（5推定器）／短調は未** ← ここが本命の欠け | `results/reanalysis/a9/sr_by_estimator.csv`（KK 0.3555 / Aarden 0.4191 / Bellman 0.5682 / TKP 0.5464 / Viterbi 0.3736、いずれも edit と k1_norm の2条件、**長調のみ 2,200行**） |
| **E**（steering 比較） | **一部済** — 最終テストで B（変位一致）0.2445、C（α=2）0.4709、D（対比）0.3864。さらに探索プロンプトで α グリッド全点の per-row（擾乱付き） | `results/steering/R-Aug_s0/final_verdict{,_minor}.json`、`parts/C_L*_a*.parquet` |
| E の欠け | ①最終テストでの α 細分（1.25/1.5/1.75/2.5/3.0）②**置換のスケール版**（`h + s(−P_V h + P_V µ)`）は編集モードとして未実装 ③同効果点の補間比較 | ②は `SubspaceEditor` に新モードを1つ追加（加算的変更） |
| **C1**（生成前シフト・公開モデル） | **AMT-small × Bach は済** | 補足 §24、`results/reanalysis/a11/next_pitch_music-small-800k_L8.parquet`（−0.0239 → −0.0096、p_Holm 0.0018、ランダム対照は不動） |
| C1 の欠け | REMI と MMT | 既存スクリプトのモデル差し替え（前向き計算のみ） |
| **C2**（位置対照・公開モデル） | **未実装**（コード自体が無い） | `HookSubspaceEditor` に `token_mask` が無く、公開スキームのトークン種別同定も無い |
| **C3**（AMT 層スイープ） | **概ね素材あり** | `results/mwild_sweep/music-small-800k/stage1_layer_scan.json`（編集側）＋ `results/mwild/*/mwild_probe.json`（probe 側の層別 F1） |
| **A**（ピッチクラス対照部分空間） | **完全に新規**（最重要） | 該当 artifact 無し |

→ 方針: **既存で答えが出ている部分は再計算せず、artifact を引用して表にまとめる**。
指示書の書式（`tables/<exp>.md`）は満たしつつ、出所を明記する。

---

## 3. 未実装項目（§0.1 の要求により明示）

1. **自然転調時 perplexity 変化の per-event 値が未保存**（実験 B1）
   `freeze_quality_guard.py:56` が 7,989 件のリストを即座に縮約し、保存されるのは
   要約 `rise_distribution = {mean, std, p50, p75, p90, p99}` のみ。したがって
   **p80/p85/p95 は artifact から復元できない**。入力（`M-REF_s100/final.pt`、
   `results/data_syn/val.parquet`）は残っているので再計算は可能（前向き計算のみ）。
   → 対処: 新パス `results/guard/rise_distribution_recomputed.json` に書き出す
   （凍結済み `delta_ppl.json` は**絶対に上書きしない**）。再計算した p90 が
   0.6127123 を再現することを整合性チェックとして先に確認する。
2. **ridge 回帰ユーティリティが無い**（実験 A2）
   リポジトリ全体に `Ridge`/`lstsq`/`pinv`/`LinearRegression` が存在しない。既存の
   `train_probe` は 24 クラス分類器（cross-entropy）なので流用不可。
   → 対処: `numpy.linalg.solve` による閉形式 ridge を新規に書く（20行程度）。
   再利用できるのは `probes.py:20` `split_by_sequence`（系列単位分割）と標準化処理。
   なお**窓つきピッチクラスヒストグラムは既存**（`src/probing/extract.py:79`
   `pc_hist_window(ids, t, w)`、`extract(..., windows=[...])` が `pc_hist_W{w}` 列を出す）。
3. **公開モデルのトークン種別マスクが無い**（実験 C2）
   `HookSubspaceEditor` は `token_mask` を持たず、`src/publicmodels/` 側にトークン
   種別（音高／小節・長さ）の同定ロジックも無い。実装には (i) フックエディタへの
   マスク追加、(ii) 各スキームの種別定義（AMT: note 対 time+duration、REMI: Pitch 対
   Bar+Position+Duration、MMT: pitch フィールド）、(iii) 単体テストが必要。
   → **指示書 §5 の要求どおり共通インタフェース＋単体テストで実装する**が、
   これは新規実装なので着手前に確認を取る（下記 §5 Q3）。
4. **短調の継続に対する推定器差し替えが未実行**（実験 D3）
   トークン列は `results/rescore/continuations_R-Aug_s0_minor_L4.json` に存在するので
   **生成不要・CPU のみの再解析で埋まる**。D の中で最も価値の高い欠けであり、
   「短調の優位は推定器の産物か」に直接答える。

---

## 4. 計算量の見積り（§0.4）

スループットは台帳の実測から **115 continuations/分**（自前25Mモデル、バッチ64、
384トークン上限。REPRO.md §3）。1条件 = 12標的 × 100プロンプト = 1,200 continuations。

| 実験 | 新規生成 | 見積り | 1時間超か |
|---|---|---|---|
| B | 0（再解析）＋ guard 再計算（前向きのみ） | 20–40 分 | いいえ |
| D | 0（再解析。短調の推定器差し替えは CPU） | 20–30 分 | いいえ |
| **A**（指示書どおり: 3条件 + ランダム対照3×3 = 12条件 × 2 modes） | 28,800 | **約 4.2 時間** | **はい** |
| A（縮約案: ランダム対照を各1個 = 6条件 × 2 modes） | 14,400 | 約 2.1 時間 | はい |
| A（さらに縮約: 長調のみ・対照各1個） | 7,200 | 約 1.0 時間 | 境界 |
| **E**（steering 新6点 + 置換スケール新4点 = 10点 × 2 modes） | 24,000 | **約 3.5 時間** | **はい** |
| E（長調のみ全点） | 12,000 | 約 1.7 時間 | はい |
| E（探索プロンプトの既存 α グリッドを流用する版） | 0 | 30 分（再解析） | いいえ |
| **C1**（REMI, MMT の生成前シフト） | 前向き計算のみ、各 60×12×3 | 各 15–30 分（要実測） | 個別には いいえ |
| **C2**（位置対照 2モデル × 3条件） | 公開モデル生成、要 dry-run 実測 | **未見積り**（実装後に dry-run で報告） | おそらく はい |
| C3（AMT 層スイープ） | 既存 artifact 中心 | 30 分 | いいえ |

締切は 2026-09-16 AoE（残り7日）。本文は現在6ページで 4+1 規定を超過しているため、
**追加結果の受け皿は補足資料**（既存の再解析節と同じ扱い）を前提に計画している。

---

## 5. 確認したいこと（作業を止めて質問。§0.6, §5）

**Q1. A と E の実行規模** — どちらも1時間を超えるため §0.4 により確認が必要です。
推奨は「A はランダム対照を各1個に縮約（約2.1時間、長調・短調とも実施）」＋
「E は既存の探索段階 α グリッドで Pareto 曲線を先に作り（再解析・0時間）、
必要と判断できたら最終テストの新点を長調のみ追加（約1.7時間）」。この配分で
進めてよいでしょうか。フル規模（A 4.2h + E 3.5h）でも構いません。

**Q2. 既存成果の扱い** — B・D・E・C1 の既済部分は再計算せず、artifact を引用して
指示書の表形式にまとめる方針で進めます（同じ生成を2度走らせないため）。
「欠けている部分」（B の他アーム／steering／公開モデル、**D3 の短調**、C1 の
REMI・MMT）だけを新規に埋めます。異論があれば教えてください。

**Q3. C2（公開モデルの位置対照）の実装** — トークン種別マスクが公開モデル側に
無いため、フックエディタへのマスク追加＋スキーム別の種別定義＋単体テストという
新規実装が必要です（C の中で最も重い）。実装に進めますか、それとも C は
C1（REMI/MMT の生成前シフト）と C3（層スイープ）に絞りますか。

**Q4. 事前登録** — このプロジェクトの流儀（`docs/CONFIRMATORY_FREEZE.md` 等）に
合わせ、A〜E の設計と「結果の読み方」を走らせる前に
`docs/ADDITIONAL_EXPERIMENTS_FREEZE.md` としてコミットすることを推奨します。
指示書の「結果の読み方」（例: 2 ≪ 1 かつ 4 ≈ 1 → 主張が強まる／2 ≈ 1 → 弱まるが
そのまま報告）が既に事前登録の体裁なので、それをそのまま凍結文書にします。
30分程度で、査読上の資産（事前登録の一貫性）を守れます。進めてよいでしょうか。

---

## 5b. 実施状況（2026-09-10 更新）

| 実験 | 状態 | artifact | 書き足し先 |
|---|---|---|---|
| B 閾値感度 | **完了**（再解析、生成ゼロ） | `results/reanalysis/b_threshold/` | 補足 §4 + 表 S14 |
| D1 mode別 in-key | 既存 | `reanalysis/a5/minor_handling.json` | 補足 §18 |
| D2 着地・相対長調 | **完了**（短調を表に拡張） | `reanalysis/a2/landing.json` | 補足 §11 表 S9 |
| D3 代替推定器 長調 | 既存 | `reanalysis/a9/` | 補足 §17 |
| D3 代替推定器 短調 | **完了**（CPUのみ） | `reanalysis/a9_minor/` | 補足 §17（主張を1つ狭めた） |
| A ピッチクラス対照 | **完了**（事前登録 c68ae44） | `reanalysis/a_pitchclass/` | 補足 §13 表 S16 + 本文 §4.4 |
| E Pareto（探索段階） | **完了**（生成ゼロ） | `reanalysis/e_pareto/` | 補足 §7 表 S15 |
| E2 スケール版据え付け | **実行中** | `reanalysis/e2_scaled/` | 未 |
| C1 生成前シフト ×3 | **完了**（AMENDMENT 1） | `reanalysis/c1_public_next_pitch/` | 補足 §24 |
| C2 位置対照（公開） | 実装完了・実行待ち | — | 未 |
| C3 層スイープ（AMT） | 未着手（既存 artifact で可能） | `mwild_sweep/*/stage1_layer_scan.json` | 未 |

実装として新規に入ったもの（いずれも既存挙動は不変、テストで固定）:
- `SubspaceEditor` の `replace_scaled` モード（s=1 で replace とビット一致、
  s=0 で恒等。tests/test_replace_scaled.py の4件）
- `HookSubspaceEditor` の `token_mask` / `mask_kind`（全Trueでマスク無しと一致）
- アダプタ契約の任意メソッド2つ: `next_pitch_class_mass`（C1）と
  `token_type_mask`（C2、複合方式は拒否）。tests/test_public_token_masks.py の6件
- `public_edit_sweep.py` の `--positions`（フラグ無しなら従来と同一）

## 6. ハイパーパラメータの決定過程（§0.2、決まり次第追記）

| 対象 | 候補 | 決定手続き | 決定値 |
|---|---|---|---|
| 実験A ridge の λ | {1e-3, 1e-2, 1e-1} | **探索用100プロンプト**（test rows 0–167）で選ぶ。最終テストのプロンプト（rows 6000–）は回帰・λ選択・窓幅選択に一切使わない | **1e-3**（探索プロンプト上の平均 R² 0.718 で最良。他は 0.701 と 0.643） |
| 実験A 窓幅 w | 論文で最強だった note-counting の窓 ＋ 全履歴 | 最強窓は既に確定済み: `lr_W16cat512`（W16 と W512 の連結、F1 0.824）。したがって短窓 w=16、全履歴 w=512 を採用 | w ∈ {16, 512} |
| 実験A V_pc の次元 | 12 / 24 | 指示書どおり両方（V_pc12 = 全履歴12行、V_pc24 = 短窓12 + 全履歴12） | 12, 24 |
| 実験E の s グリッド | 指示書の指定どおり | 探索段階の既存 α グリッド（0.25–16）から Pareto 上の関心領域を確認して確定 | steering 側は既存の α∈{0.25..16}（探索段階）＋最終テストの3点で代替。置換のスケール版は **s∈{0.5,0.75,1.25,1.5}**（s=1 は凍結行を再利用）を AMENDMENT 1 で固定 |
