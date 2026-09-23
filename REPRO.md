# REPRO.md — 本文主要値の再現確認

当初は「追加実験を走らせる前に、旧 Table 1 の主要値が ±0.01 以内で再現すること」
を確認するための記録だった。現在の本文では、主モデルの主要値は Table ではなく
Fig. 2 に置かれ、Table 1 は公開モデルの結果をまとめる表になっている。
このファイルは、現在の Fig. 2 と関連する本文主要値が artifact から再計算できることの
確認記録として読む。

## 1. 再現確認の結果（全項目 合格）

台帳済み artifact から丸めを再計算した値と、現在の本文記載値の比較。

| 量 | 論文 | artifact 実測 | 差 | 判定 |
|---|---|---|---|---|
| Fig. 2左: 置換 SR（長調） | 0.355 | 0.3555 | +0.0005 | OK |
| Fig. 2左: 置換 SR（短調） | 0.495 | 0.4945 | −0.0005 | OK |
| Fig. 2左: 変位一致ランダム対照（長調） | 0.056 | 0.0564 | +0.0004 | OK |
| Fig. 2左: 変位一致ランダム対照（短調） | 0.026 | 0.0264 | +0.0004 | OK |
| Fig. 2右: 置換 $\delta D$（長調） | 0.695 | 0.6953 | +0.0003 | OK |
| Fig. 2右: 置換 $\delta D$（短調） | 0.309 | 0.3094 | +0.0004 | OK |
| Fig. 2右: ランダム対照 $\delta D$（長調） | 0.041 | 0.0410 | +0.0000 | OK |
| Fig. 2右: ランダム対照 $\delta D$（短調） | 0.026 | 0.0261 | +0.0001 | OK |
| 尤度基準なし SR（長調） | 0.410 | 0.4100 | +0.0000 | OK |
| 尤度基準なし SR（短調） | 0.617 | 0.6173 | +0.0003 | OK |

差はすべて ±0.001 以内（要求は ±0.01）。**論文の値は3桁丸めであり、実測との差は
丸めのみに由来する**（丸め誤りの前歴があるため、値は毎回 artifact から
`f"{x:.3f}"` で再計算して文字列一致を確認している）。

根拠 artifact:
- `results/confirmatory/R-Aug_s0/verdict.json` — `conditions.edit.pooled_guarded_tkr`,
  `edit_vs_k1norm.pooled_k1_norm`, `conditions.edit.raw_tkr`
- `results/confirmatory/R-Aug_s0_minor/verdict.json` — 同上（短調）
- `results/confirmatory/R-Aug_s0/next_pitch.json`,
  `results/confirmatory/R-Aug_s0_minor/next_pitch.json` — `pooled.mean_D_edit`,
  `pooled.mean_D_k1`

再現コマンド（本ファイルの表を生成したもの）:

```bash
cd <REPO_ROOT>   # = the directory this repository is checked out into
.venv/bin/python - <<'EOF'
import json
for mode, sub in (("major","R-Aug_s0"), ("minor","R-Aug_s0_minor")):
    v = json.load(open(f"results/confirmatory/{sub}/verdict.json"))
    n = json.load(open(f"results/confirmatory/{sub}/next_pitch.json"))
    print(mode,
          f"{v['conditions']['edit']['pooled_guarded_tkr']:.4f}",
          f"{v['edit_vs_k1norm']['pooled_k1_norm']:.4f}",
          f"{v['conditions']['edit']['raw_tkr']:.4f}",
          f"{n['pooled']['mean_D_edit']:.4f}",
          f"{n['pooled']['mean_D_k1']:.4f}")
EOF
```

既存の自動照合器も併走させている（数値が .tex から artifact へ辿れるかの検査）:

```bash
.venv/bin/python experiments/figures/collect_paper_numbers.py --check-tex
# expected: untraceable across both documents: 0
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
| テスト | `pytest tests/ -q` → 393 passed（確認時。テストは以後も追加された） |

指示書の `<GPU_SPEC>` は未置換だったため、実機の上記 GPU を前提として見積もった。
`<REPO_ROOT>` も未置換だったため、チェックアウト先のディレクトリと解釈した。

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
