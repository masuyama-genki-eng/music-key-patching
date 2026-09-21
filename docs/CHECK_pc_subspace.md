# CHECK pc subspace

範囲: 確認のみ。コード変更、実験再実行、生成、TeX編集は行っていない。保存済み artifact と台帳、該当スクリプトを読んだ。SVD値だけは保存済み `.npz` 内の行列から再計算した。

## 1. 実装と台帳ID

| 対象 | 実装 | 主な関数 / 箇所 | artifact | 台帳ID |
|---|---|---|---|---|
| 主モデル pc subspace fit | `experiments/reanalysis/pitchclass_subspace.py` | `ridge`, `orthonormal`, `stage_fit` | `results/reanalysis/a_pitchclass/subspaces.npz`, `results/reanalysis/a_pitchclass/overlap.json` | `2026-09-09T18:35:27+09:00` / `2026-09-09T18:36:48+09:00`, `EXP A stage 1: pitch-class regression subspaces` |
| 主モデル pc/res edit | `experiments/reanalysis/pitchclass_edit.py` | `main`, `SUBSPACES`, `CTRL_SEED_OFFSET` | `results/reanalysis/a_pitchclass/edit_rows_{major,minor}.parquet`, `verdict_{major,minor}.json` | major: `2026-09-09T19:45:14+09:00`; minor: `2026-09-09T20:49:58+09:00` |
| 公開モデル pc subspace fit | `experiments/public_models/public_pitchclass_subspace.py` | `ridge`, `hists`, `orthonormal`, `main` | `results/public_pc_control/<ckpt>/pc_fit.json`, `<ckpt>/subspaces.npz` | AMT: `2026-09-17T23:40:56+09:00` / `23:42:33`; MMT: `2026-09-18T19:37:37+09:00`; REMI+: `2026-09-18T19:37:40+09:00` |
| 公開モデル pc/res edit | `experiments/public_models/public_edit_sweep.py` | `basis_and_means`, `run`, stage 2 scoring | `results/public_pc_control/<ckpt>/sweep_pc24/stage2_eval.json`, `sweep_res/stage2_eval.json` | pc24: AMT `2026-09-18T20:31:28`, MMT `20:58:31`, REMI+ `21:18:45`; res: AMT `22:12:32`, MMT `22:39:02`, REMI+ `22:59:13` |

## 2. 主モデルの部分空間構成

### 2.1 予測対象

| 項目 | コード上の事実 |
|---|---|
| 層 | 固定で `LAYER = 4` |
| 活性 | `h_l(t)`、`extract(...)[\"acts\"][LAYER]` |
| 窓長 | `WINDOWS = [16, 512]`。`pc_hist_window(ids, t, w)` は `ids[max(0,t-w+1):t+1]` の中の PITCH token を数える。つまり「活性位置 t までの直近 w token」。 |
| ビン数 | 12 pitch classes |
| 正規化 | `hist_features` で L1 正規化。和が0の行は分母を1にする。 |
| 入力位置数 | fit: `50400` positions。`n_fit_pieces=2100`, `per_seq=24`。lambda selection: `2400` positions。100 search prompts, `per_seq=24`。 |
| fitting data | `results/data_syn/train.parquet` の training pieces |
| lambda selection data | `results/data_syn/test.parquet` から `SW.select_prompts(..., 100)` で選ぶ search-stage prompt positions。final-test prompt は使わない。 |

### 2.2 回帰

| 項目 | コード上の事実 |
|---|---|
| 回帰 | closed-form ridge with intercept |
| 式の実装 | `A = X.T @ X + lam * n * I`; `B = X.T @ (Y - Y.mean)`; `W = solve(A, B).T`; intercept `b = Y.mean` |
| 標準化 | fit activations は `Z = (X - mu) / sd` に標準化。最終 weight は `Wraw = Wz / sd` として raw activation 座標へ戻す。 |
| lambda grid | `[1e-3, 1e-2, 1e-1]` |
| 選び方 | search prompt positions 上の2窓平均 R2 最大。選択値は `lambda=0.001`。 |
| R2 | `lambda=0.001`: W16 search R2 `0.6544`, W512 search R2 `0.7811`, mean `0.7178`。 |

### 2.3 重み行列・特異値・ランク

ランク判定は `SVD_TOL = 1e-6`、つまり `S > 1e-6 * S[0]`。

| 行列 | 形状 | 特異値 | 数値ランク |
|---|---:|---|---:|
| `W_pc16` | `12 x 512` | `[0.0777915, 0.0742332, 0.0717566, 0.0679557, 0.0630474, 0.0573317, 0.0567950, 0.0510722, 0.0464506, 0.0334753, 0.0329839, 1.82e-15]` | 11 |
| `W_pc512` | `12 x 512` | `[0.0670053, 0.0648805, 0.0554539, 0.0539014, 0.0423325, 0.0395723, 0.0373467, 0.0348000, 0.0250429, 0.0160834, 0.0155901, 1.42e-15]` | 11 |
| stacked `[W_pc16; W_pc512]` | `24 x 512` | `[0.0841439, 0.0801002, 0.0783893, 0.0749838, 0.0728940, 0.0706019, 0.0617509, 0.0555713, 0.0504974, 0.0473200, 0.0470181, 0.0455051, 0.0420811, 0.0368805, 0.0348532, 0.0342375, 0.0330684, 0.0300122, 0.0282536, 0.0212455, 0.0119504, 0.0111885, 1.76e-15, 1.41e-15]` | 22 |

12-bin L1 histogramなので各窓の重みは実質 rank 11。2窓を stack した 24-row 行列は rank 22。

### 2.4 基底の作り方

| 基底 | 作り方 | rank |
|---|---|---:|
| `V_pc12` | `orthonormal(Wraw[512])`。`W_pc512` の row space の直交基底。 | 11 |
| `V_pc24` | `orthonormal(vstack([Wraw[16], Wraw[512]]))`。stacked weights の row space の直交基底。 | 22 |
| `V_res` | `V - V_pc24 @ (V_pc24.T @ V)` で `V_l` から pc 部分空間を射影で除き、その row/column span を `orthonormal(resid.T)` で直交化。 | 24 |

重要: 「`V_l` とランクを揃えるためにランダム方向を足す」操作は存在しない。24行を stack するが、SVDで rank 22 と判定されたらそのまま rank 22 の `V_pc24` を使う。残差 `V_res` は射影後も rank 24 のまま。

## 3. 主モデル SR_pc のランダム対照

`pitchclass_edit.py` では、各 arm に対して `SW.k1_basis(V, seed)` を使う。

| arm | rank | random seed | 変位一致 | 実装 |
|---|---:|---:|---|---|
| `pc24_rand` | 22 | `confirm.GEN_SEED + 31*LAYER + 1 = 7 + 124 + 1 = 132` | あり | `SubspaceEditor(..., norm_ref=Vt)`。random basis で編集するが、delta norm は pc24 basis の edit に合わせる。 |
| `pc12_rand` | 11 | `133` | あり | 同上 |
| `res_rand` | 24 | `134` | あり | 同上 |

したがって、主モデルの `pc24_rand` は「同じ rank かつ displacement-matched random subspace」で合っている。

## 4. 残差アーム

主モデルも公開モデルも、残差は

`V_res = orthonormal((V - P_pc V).T)`

で作る。主モデル artifact では `V_res` の形状は `512 x 24`、rank は 24。公開モデルも AMT `768 x 24`、MMT/REMI+ `512 x 24`、rank は全て 24。pc 部分空間を除いても `V_l` の残りは rank 24 のまま。

## 5. 公開モデル Bach 3モデルでの手順

公開モデルは同じ考え方だが、窓の単位だけ違う。自作モデルの 16/512 token に対応させるため、公開モデルでは `7/222` preceding note events を使う。

| model | layer | fit positions | selection positions | windows | 正規化 | lambda | rank `V_pc24` | rank `V_res` |
|---|---:|---:|---:|---|---|---:|---:|---:|
| AMT-12L | 8 | 2200 | 200 | 7 / 222 preceding note events | L1 | 0.1 | 22 | 24 |
| MMT | 5 | 2200 | 200 | 7 / 222 preceding note events | L1 | 0.1 | 22 | 24 |
| REMI+ | 5 | 2200 | 200 | 7 / 222 preceding note events | L1 | 0.01 | 22 | 24 |

公開モデルの `hists(pitch_windows, w)` は `ps[:-1][-w:]` を使うため、予測される pitch 自身を除外し、直前の pitch events だけから 12-bin histogram を作る。

公開モデルの weight singular values:

| model | 行列 | 形状 | 特異値 | rank |
|---|---|---:|---|---:|
| AMT-12L | `W_pc7` | `12 x 768` | `[0.358491, 0.341829, 0.328394, 0.311733, 0.303885, 0.273173, 0.257186, 0.223505, 0.212827, 0.194164, 0.176675, 2.56e-15]` | 11 |
| AMT-12L | `W_pc222` | `12 x 768` | `[0.319912, 0.294929, 0.265832, 0.240779, 0.151502, 0.147407, 0.131885, 0.121655, 0.103847, 0.0807845, 0.0672000, 1.55e-15]` | 11 |
| AMT-12L | stack | `24 x 768` | `[0.376553, 0.363047, 0.341613, 0.326725, 0.311604, 0.298602, 0.287343, 0.275465, 0.267429, 0.233078, 0.223248, 0.219294, 0.209489, 0.194860, 0.177709, 0.136143, 0.133320, 0.123696, 0.117224, 0.100084, 0.0763615, 0.0645657, 2.56e-15, 1.52e-15]` | 22 |
| MMT | `W_pc7` | `12 x 512` | `[0.0134968, 0.0128008, 0.0124549, 0.0115412, 0.0109423, 0.0100354, 0.00965037, 0.00874216, 0.00834341, 0.00754921, 0.00683627, 5.43e-17]` | 11 |
| MMT | `W_pc222` | `12 x 512` | `[0.0103628, 0.00873294, 0.00779858, 0.00720633, 0.00435229, 0.00416096, 0.00374773, 0.00345884, 0.00299759, 0.00226998, 0.00174414, 4.07e-17]` | 11 |
| MMT | stack | `24 x 512` | `[0.0142617, 0.0134205, 0.0126875, 0.0118732, 0.0113292, 0.0103744, 0.0101665, 0.00936969, 0.00892927, 0.00845139, 0.00785418, 0.00733545, 0.00691836, 0.00661842, 0.00629669, 0.00385026, 0.00366905, 0.00329511, 0.00308972, 0.00272335, 0.00204745, 0.00163715, 5.26e-17, 4.04e-17]` | 22 |
| REMI+ | `W_pc7` | `12 x 512` | `[0.0327067, 0.0307653, 0.0266415, 0.0251689, 0.0238877, 0.0233205, 0.0214031, 0.0187195, 0.0164130, 0.0149727, 0.0139562, 4.03e-16]` | 11 |
| REMI+ | `W_pc222` | `12 x 512` | `[0.0211710, 0.0185941, 0.0171980, 0.0148379, 0.0110158, 0.0105043, 0.00909684, 0.00852579, 0.00744404, 0.00591443, 0.00492284, 3.37e-16]` | 11 |
| REMI+ | stack | `24 x 512` | `[0.0328988, 0.0308880, 0.0270992, 0.0257897, 0.0252165, 0.0237182, 0.0219324, 0.0197815, 0.0192585, 0.0178946, 0.0168294, 0.0161455, 0.0144825, 0.0143152, 0.0134883, 0.0101338, 0.00981644, 0.00871356, 0.00817000, 0.00712273, 0.00552243, 0.00460590, 3.96e-16, 3.33e-16]` | 22 |

公開モデルのランダム対照:

| model | pc24 random rank | K1 seed | 変位一致 |
|---|---:|---:|---|
| AMT-12L | 22 | `0 + 31*8 = 248` | `sr_k1` は変位一致なし。`sr_k1_norm` は同じ random basis に norm matching あり。 |
| MMT | 22 | `0 + 31*5 = 155` | 同上 |
| REMI+ | 22 | `0 + 31*5 = 155` | 同上 |

公開モデルの `SR_pc` 表示値は `pc24.sr`。対照として `sr_k1` と `sr_k1_norm` の両方が保存されているが、`results/public_pc_control.md` の主な K1 列は displacement-matched ではない。

## 6. 本文数値との対応

| 数値 | 意味 | file | key | 台帳ID |
|---:|---|---|---|---|
| 0.027 | 主モデル major `SR_pc` | `results/reanalysis/a_pitchclass/verdict_major.json` | `arms.pc24.sr = 0.0273` | `2026-09-09T19:45:14+09:00` |
| 0.003 | 主モデル minor `SR_pc` | `results/reanalysis/a_pitchclass/verdict_minor.json` | `arms.pc24.sr = 0.0027` | `2026-09-09T20:49:58+09:00` |
| 0.360 | 主モデル major residual | `results/reanalysis/a_pitchclass/verdict_major.json` | `arms.res.sr = 0.3600` | `2026-09-09T19:45:14+09:00` |
| 0.478 | 主モデル minor residual | `results/reanalysis/a_pitchclass/verdict_minor.json` | `arms.res.sr = 0.4782` | `2026-09-09T20:49:58+09:00` |
| 0.178 | AMT-12L public `SR_pc` | `results/public_pc_control.json` | `models.AMT-12L.pc24.sr` | run `2026-09-18T20:31:28+09:00`; summary `2026-09-18T21:19:33` / `22:59:30` |
| 0.319 | MMT public `SR_pc` | `results/public_pc_control.json` | `models.MMT.pc24.sr` | run `2026-09-18T20:58:31+09:00`; summary `2026-09-18T21:19:33` / `22:59:30` |
| 0.075 | REMI+ public `SR_pc` | `results/public_pc_control.json` | `models.REMI+.pc24.sr` | run `2026-09-18T21:18:45+09:00`; summary `2026-09-18T21:19:33` / `22:59:30` |
| 0.106 | AMT-12L public residual | `results/public_pc_control.json` | `models.AMT-12L.res.sr` | run `2026-09-18T22:12:32+09:00`; summary `2026-09-18T22:59:30` |
| 0.453 | MMT public residual | `results/public_pc_control.json` | `models.MMT.res.sr` | run `2026-09-18T22:39:02+09:00`; summary `2026-09-18T22:59:30` |
| 0.567 | REMI+ public residual | `results/public_pc_control.json` | `models.REMI+.res.sr` | run `2026-09-18T22:59:13+09:00`; summary `2026-09-18T22:59:30` |

## 7. 現行本文記述との照合

照合対象:

> we fit ridge regressions from h_l(t) to the L1-normalized 12-bin pitch-class histograms of the last 16 and 512 tokens and take the row space of the weights (rank 22) as a control subspace ... a random subspace of the same rank and displacement

| 本文要素 | コード事実 | 判定 |
|---|---|---|
| `fit ridge regressions` | closed-form ridge。intercept あり。 | 一致。ただし intercept ありは本文にない。 |
| `from h_l(t)` | layer-4 activations `h_l(t)` を使う。X は標準化し、最後に raw座標の weight へ戻す。 | 一致。ただし標準化は本文にない。 |
| `to L1-normalized 12-bin pitch-class histograms` | 12-bin pitch-class count を L1 正規化。 | 一致。 |
| `last 16 and 512 tokens` | 主モデルでは `ids[max(0,t-w+1):t+1]` の PITCH token count。つまり activation position t までの 16/512 token 窓。 | 主モデルについて一致。厳密には「tを含む直近token窓」。 |
| `row space of the weights` | `W_pc16` と `W_pc512` を stack し、その row space を SVDで直交化。 | 一致。 |
| `(rank 22)` | stack weight は 24行だが、L1 histogram 制約により数値 rank 22。閾値は `1e-6*S[0]`。 | 一致。rank 22 は手動切断ではなく数値ランク。 |
| `as a control subspace` | `V_pc24` を `pc24` arm として書き込みに使う。 | 一致。 |
| `random subspace of the same rank and displacement` | 主モデル `pc24_rand` は rank 22、かつ `norm_ref=V_pc24` による displacement matching あり。 | 主モデルについて一致。 |
| 公開モデルにも同じ文を適用する場合 | 公開モデルでは窓は 16/512 token ではなく 7/222 preceding note events。`SR_pc` の K1 対照は rank matched だが displacement matched ではない。K1-norm は別に保存されている。 | 公開モデルにそのまま書くなら不一致。 |

## 8. 修正文案

主モデルだけを述べるなら、現行文はほぼ正しいが、次のようにするとコード事実により忠実。

> For the synthetic-corpus model, we fit ridge regressions with an intercept from standardized layer-4 activations to L1-normalized 12-bin pitch-class histograms over the 16- and 512-token windows ending at the activation position. We stack the two 12-row weight matrices and use their numerical row space as the pitch-class control subspace; its rank is 22 under the SVD threshold `1e-6 S_max`. Its random control has the same rank and is displacement-matched.

公開モデルも同じ段落で述べるなら、次の補足が必要。

> For the public Bach models, the analogous windows are 7 and 222 preceding note events, chosen to match the synthetic tokenizer's 16- and 512-token windows. The reported public `SR_pc` values use the rank-matched K1 control; the displacement-matched K1-norm control is stored separately.

結論: 主モデルの本文記述は大筋で一致。ただし、intercept/standardization、rank 22 が数値ランクであること、公開モデルでは 7/222 note-event windows であり `SR_pc` の主対照が K1 で displacement-matched ではないことは、誤解を避けるなら明記した方がよい。
