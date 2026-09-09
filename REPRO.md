# REPRO.md — 追加実験 A〜E の前提となる再現確認

指示書 §0.3 の要求「新条件を走らせる前に、既存の Table 1 主要値が ±0.01 以内で
再現することを確認」に対する記録。2026-09-09 実施。

## 1. 再現確認の結果（全項目 合格）

台帳済み artifact から丸めを再計算した値と、論文 Table 1 の記載値の比較。

| 量 | 論文 | artifact 実測 | 差 | 判定 |
|---|---|---|---|---|
| 置換 SR（長調） | 0.355 | 0.3555 | +0.0005 | OK |
| 置換 SR（短調） | 0.495 | 0.4945 | −0.0005 | OK |
| 変位一致ランダム対照（長調） | 0.056 | 0.0564 | +0.0004 | OK |
| 変位一致ランダム対照（短調） | 0.026 | 0.0264 | +0.0004 | OK |
| 移調参照（長調） | 0.648 | 0.6482 | +0.0002 | OK |
| 移調参照（短調） | 0.877 | 0.8773 | +0.0003 | OK |

差はすべて ±0.001 以内（要求は ±0.01）。**論文の値は3桁丸めであり、実測との差は
丸めのみに由来する**（丸め誤りの前歴があるため、値は毎回 artifact から
`f"{x:.3f}"` で再計算して文字列一致を確認している）。

根拠 artifact:
- `results/confirmatory/R-Aug_s0/verdict.json` — `conditions.edit.pooled_guarded_tkr`,
  `edit_vs_k1norm.pooled_k1_norm`
- `results/confirmatory/R-Aug_s0_minor/verdict.json` — 同上（短調）
- `results/confirmatory/R-Aug_s0/k4_ceiling.json`,
  `results/confirmatory/R-Aug_s0_minor/k4_ceiling.json` — `k4_raw_tkr_nonidentity`

再現コマンド（本ファイルの表を生成したもの）:

```bash
cd <REPO_ROOT>   # = /home/masuyama-genki/ICASSP③/tonal-world-model
.venv/bin/python - <<'EOF'
import json
for mode, sub in (("major","R-Aug_s0"), ("minor","R-Aug_s0_minor")):
    v = json.load(open(f"results/confirmatory/{sub}/verdict.json"))
    k4 = json.load(open(f"results/confirmatory/{sub}/k4_ceiling.json"))
    print(mode, f"{v['conditions']['edit']['pooled_guarded_tkr']:.4f}",
          f"{v['edit_vs_k1norm']['pooled_k1_norm']:.4f}",
          f"{k4['k4_raw_tkr_nonidentity']:.4f}")
EOF
```

既存の自動照合器も併走させている（数値が .tex から artifact へ辿れるかの検査）:

```bash
.venv/bin/python experiments/figures/collect_paper_numbers.py --check-tex
# -> untraceable across both documents: 0   (2026-09-09 時点)
```

さらに、パイプライン自体の非退行はリポジトリ側の回帰ゲートが担保している
（`experiments/steering/steering_regression.py`: 見出し値の再計算、sham 編集の
ビット同一性、探索プロンプトのクリーン継続がトークン単位で一致するか）。
`results/steering/R-Aug_s0/regression.json` は `passed: true`。

## 2. 環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 48 GB（49,140 MiB）, driver 580.173.02 |
| Python | 3.12.3（`.venv`） |
| PyTorch | 2.13.0+cu130 / CUDA 13.0 |
| リポジトリ HEAD（確認時） | `3e53bc7` |
| テスト | `pytest tests/ -q` → 365 passed |

指示書の `<GPU_SPEC>` は未置換だったため、実機の上記 GPU を前提として見積もった。
`<REPO_ROOT>` も未置換だったため `/home/masuyama-genki/ICASSP③/tonal-world-model`
と解釈した。

## 3. 生成スループット（見積りの根拠）

台帳の実測から: 短調の最終テスト（5アーム × 12標的 × 100プロンプト = 6,000
continuations）は K2 ゲート通過 01:08 の 52 分前に開始しており、
**約 115 continuations/分**（自前 25M モデル、バッチ 64、384 トークン上限）。

この実測値で追加実験の所要時間を見積もる（詳細は PLAN.md §4）。

## 4. seeds

| 用途 | seed | 固定場所 |
|---|---|---|
| 生成（最終テスト） | 7 | `experiments/confirmatory/confirmatory_test.py` `GEN_SEED` |
| ランダム対照基底 | `GEN_SEED + 31*layer` | `src/intervene/sweep.py:161` `k1_basis` |
| bootstrap | 固定（10,000 反復） | `src/analysis/stats.py` `bca_ci` |
| 追加実験で新たに引く乱数 | 実験ごとに記録 | `results/<exp>/records.jsonl` の `seed` 欄 |
