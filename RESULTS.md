# RESULTS.md — 追加実験 A〜E の結果

指示書 §1.5 と §4 の成果物。各実験について「1段落の日本語要約」「論文に挿入できる
英文」「表・図への参照」を置く。数値はすべて台帳済み artifact から
`experiments/figures/collect_paper_numbers.py` が再計算したもので、手入力はない
（`--check-tex` は両文書で untraceable 0 を維持）。

事前登録: 実験A は `docs/ADDITIONAL_EXPERIMENTS_FREEZE.md`（commit c68ae44、
artifact ゼロの時点）、C1・E2・C2 は同文書の AMENDMENT 1（commit 61ba8de、同様）。
B・D・E(探索段階) は保存済みの行を採点し直すだけで設計判断を含まないため事前登録は不要。

---

## 実験A — ピッチクラス分布対照部分空間

**要約**: この論文で最も鋭い反論——「編集している V は直前音のピッチクラス頻度の
線形像にすぎない」——は成り立たなかった。層4の活性からピッチクラスヒストグラムへの
リッジ回帰は**よく当てはまる**（探索プロンプト上 R² は短窓 0.654、全履歴 0.781）のに、
その部分空間は V とほとんど重ならない（重なり 0.056 対 ランダム24次元の帰無
0.047±0.003、主角22本すべて 63.55°–89.40°、30°未満は0本）。因果的にも、同じ標的値を
ピッチクラス部分空間経由で書き込むと長調 0.0273・短調 0.0027 にとどまり、**同次元・
同変位のランダム部分空間と区別できない（0/12）**。逆に V からその部分空間を射影除去
した残差で書き込むと長調 0.3600・短調 0.4782 に達し、据え付けとの対応差は両旋法で
信頼区間がゼロを含む。事前に固定した3つの読み方のうち「主張が強まる」に該当した。

**英文（本文 §4.4 に挿入済み）**:
> The subspace we edit is also not the note counts in other coordinates.
> A regression from the same activations to the recent pitch-class frequencies fits well at $R^2$ $0.78$, yet writing the same value through the subspace it spans reaches only $0.027$ and is indistinguishable from a random subspace of the same dimension and displacement, while removing that subspace from $V$ still reaches $0.360$.

**参照**: 補足 §13（表 S16）。artifact `results/reanalysis/a_pitchclass/`。

---

## 実験B — 擾乱閾値の感度

**要約**: 閾値 0.613 の選択に結果は依存しない。保存済みの行を τ=0.2 から限度なしまで
再採点しても、**条件の順位は一度も入れ替わらない**（自前5アーム×2旋法、steering 3条件
×2旋法、公開8セル）。凍結閾値での順位との Spearman は τ=0.2 以外の全点で 1.000
（τ=0.2 では対照同士が入れ替わり 長調 0.900・短調 0.975、置換の首位は不動）。
副産物が2つある。限度なしの値が既存 artifact の `raw_tkr`（0.410 / 0.6173）と一致して
再採点コードの検算になった。そして**音高位置のみは全閾値で一定**（0.035 / 0.000）で、
音楽をほとんど変えないため予算が効かない——位置の帰無結果は採点規則の産物ではない。
不都合な点も出た: 2倍変位の加算は向きが閾値に依存し、短調では凍結閾値付近で交差する。

**英文（補足に記載済み。本文に足すなら）**:
> Re-scoring every condition from a threshold of $0.2$ out to no limit at all leaves the ordering unchanged, and the pitch-position null is flat across every threshold because that write never pays the budget.

**参照**: 補足 §4（表 S14）。artifact `results/reanalysis/b_threshold/`。

---

## 実験C — 公開チェックポイント

### C1 生成前の次音シフト（3モデル）

**要約**: サンプリングなしの前向き計算1回で、編集は次音の対数比を無編集より大きく
動かし、REMI と MMT では**符号ごと反転**させた（AMT −0.0239→−0.0096、REMI
−0.1048→+0.5171、MMT −0.6895→+0.2051）。ランダム部分空間との対応比較は
+0.0119（r 0.505）、+0.5955（r 0.995）、+0.8197（r 1.000）。REMI と MMT では対照も
無編集比でわずかに動くので、凍結した読み方が言う「対比」を測る直接比較を後から
追加した（既存の条件別検定は不変）。差は20〜30倍で、対照の小さな動きは編集の効果を
説明しない。実施には2つの実装欠陥の修復が必要で、修復後に AMT の公表値が完全再現
することを確認してから他2つを走らせた。

**英文（補足に記載済み）**:
> The edit moves the next-pitch log ratio further than a rank-matched random subspace in every edited public checkpoint, by $+0.012$, $+0.596$ and $+0.820$, and in REMI and MMT it reverses the ratio's sign before a single note is drawn.

**参照**: 補足 §24。artifact `results/reanalysis/c1_public_next_pitch/`。

### C3 層をまたいだ read–use gap

**要約**: 生成ゼロ。層別プローブと層スキャンを同じ軸に載せると、AMT-small では
編集効果の頂点が第8層（+0.287）、プローブとマージンの頂点は第10層で、**両端に
「読めるのに効かない」層がある**——第0層はマージン +0.124（第7層の +0.127 と同等）で
編集効果 −0.004、第11層はマージン +0.188 で編集効果 +0.029。これは本文図2の弱点を
埋める: 我々のモデルではマージンが編集効果と並走するため乖離は「生の精度」について
のものだったが、公開モデルでは**統制した指標そのもの**が編集の効く層を特定できない。
編集側は1点240継続なので図に Wilson 区間を描き、隣接層の細かい差は読まない。

**英文（本文 §4.5 に挿入済み）**:
> The layer-wise separation also appears in a public checkpoint, and in a stronger form than in our own model.
> Across AMT's $12$ layers the edit gain peaks at layer~$8$ while the probe margin peaks at layer~$10$, and at layers~$0$ and~$11$ the margin is high while the edit gain sits at the floor, so here the margin over the note counts fails to locate the layers where the edit works.

**参照**: 補足 §25（図 S9b）。artifact `results/reanalysis/c3_layer_gap/`。

### C2 位置対照（公開モデル）

**要約**: 位置の非対称は我々のトークン化に固有ではない。AMT では非対称がそのまま
再現し、音高位置のみは 0.0597（ランダム対照 0.0639、0/12）、時刻と音長の位置のみは
0.3514 で制限なしの編集 0.3653 の **96%** を運び 11/12 で有意。REMI では最初どちらの
制限も無効（0.0694 と 0.0611、ともに 0/12、制限なしは 0.4208）で、凍結した読み方では
null だった。しかし REMI の2マスクは位置の **75.9%** しか覆っておらず、抜けていたのは
**instrument トークン**——REMI のプローブが調を読む位置そのもの——だった。そこで
AMENDMENT 2 として「null を見た後の追加条件」と明記して事前登録し実行すると、
**instrument 位置のみで 0.4306、制限なしの 102% を再現し 11/12 で有意**。音階内割合も
0.881 で制限なしと同等（他の2条件は 0.58 台＝無編集水準）。3つのトークン化に共通するのは
**音高トークンそのものでは効かない**ことであり、効く位置はいずれも次の音高が決まる
直前の構造トークンである（我々のモデルは小節・音長、AMT は時刻・音長、REMI は楽器）。

**英文（本文 §4.5 に挿入済み）**:
> The position result reproduces in two public schemes as well, where writing at the pitch token itself is inert in both and the effect sits instead on the structural token before a pitch is chosen, the time and duration tokens in AMT and the instrument token in REMI.

**参照**: 補足 §26（表 S18）。artifact `results/mwild_sweep/{music-small-800k/stage2_eval_positions.json,remi-lmd-remi/stage2_eval_positions3.json}`。実装は
`HookSubspaceEditor` の `token_mask`／`mask_kind` と、アダプタの `token_type_mask`
（AMT は note 対 time+duration、REMI は pitch 対 beat+position+duration、**MMT は
複合方式なので拒否**——これ自体が結果）。単体テスト6件で、2集合の排他性・全Trueマスクが
マスク無しとビット一致・複合方式の拒否を固定した。

---

## 実験D — 長調／短調差の診断

**要約**: 短調の高い成功率（0.4945 対 0.3555）は推定器のバイアスではない。5つの
推定器すべてで編集が対照を上回り（短調 0.4755–0.6236 対 対照 0.0200–0.0264）、
着地の分解も旋法固有の疑いを否定する: 編集された短調の行が相対長調に着地するのは
0.041、同主長調は 0.045 で、**どちらも統制のもとで同じ大きさ**（0.043 / 0.047）である。
一方で既存の主張を1つ狭めた。「事前登録した推定器が5つの中で最も低い率を与える」は
**長調でのみ真**で、短調では Viterbi の 0.4755 が Krumhansl–Kessler の 0.4945 より
低く、2番目に低いことになる。短調の優位そのものは推定器により +0.051 から +0.182 まで
動き、優位は残るがその大きさは測定の性質である。

**英文（補足に記載済み）**:
> The edit beats its matched control under all five estimators in minor as well, and only $0.041$ of edited minor rows land on the relative major against $0.043$ under the matched control, so the minor advantage is not the estimator confusing a key with its relative.

**参照**: 補足 §11（表 S9、両旋法）、§17（推定器）、§18（旋法差）。
artifact `results/reanalysis/{a2,a9,a9_minor,a5}/`。

---

## 実験E — steering との比較

### E(探索段階) 効果対コスト曲線

**要約**: 生成ゼロ。探索段階が α∈{0.25…16}×8層の行を残していたので、効果対コストの
曲線として読み直した。答えは層で分かれ、片側は我々に不利である。据え付けが働く第4層では
**加算は掃いたどの変位でも届かない**（最良 0.219 対 0.402）。加算自身が選んだ第2層では
変位を揃えれば据え付けが勝つ（0.276 対 0.155）が、1.353倍で追いつき2倍で追い越す。
擾乱で見ると第2層 α=2 の加算は成功率が高く擾乱中央値も低い（−0.059 対 0.162）ため、
「据え付けが効率的」は**制御量（変位）あたりであって損害量（擾乱）あたりではない**と
限定した。加算に固定点が無いことも表に現れる（α=8 で擾乱中央値 24,015 nat）。

### E2 スケール版据え付け

**要約**: 据え付けは段階的な操作である（s=1 まで単調: 長調 0.099→0.222→0.356、
短調 0.064→0.256→0.494）。しかし**我々が選んだ標的は最適ではない**: s=1.25 で
長調 0.4245・短調 0.6182、s=1.5 で 0.4573・0.6618 に達し、対応差の信頼区間はすべて
ゼロを除外する。長調の s=1.25 は擾乱中央値が s=1 より低く（0.093 対 0.104）ガード
通過率も高い（0.855 対 0.840）ので、**追加の代償なしに成功率が上がる**。なぜクラス
平均が最適でないかの説明は持っていないと明記し、凍結設計は変えず報告値は s=1 のまま
とした。事前登録の読み方のうち1と3に同時該当し、3は我々の選択に不利な結果である。

**英文（補足に記載済み）**:
> Scaling the install is monotone up to $s=1$ in both modes and keeps rising past it, to $0.4245$ at $s=1.25$ in major with a lower disturbance median than the install's, so the class-mean target is graded but not the best place to write.

**参照**: 補足 §7（表 S15 と表 S17）。
artifact `results/reanalysis/{e_pareto,e2_scaled}/`。

---

## 実装として入ったもの（すべて既存挙動は不変、テストで固定）

| 追加 | 場所 | 不変性の担保 |
|---|---|---|
| `replace_scaled` モード | `src/intervene/edit.py` | s=1 で `replace` とビット一致、s=0 で恒等（tests/test_replace_scaled.py 4件） |
| `token_mask` / `mask_kind` | `src/intervene/public_model_edit.py` | 全Trueマスクがマスク無しとビット一致 |
| `next_pitch_class_mass` | アダプタ契約（任意、既定 raise） | AMT の公表値が完全再現 |
| `token_type_mask` | アダプタ契約（任意、既定 raise） | 2集合の排他性、AMT のピッチ位置が `note_positions` と厳密一致（tests/test_public_token_masks.py 6件） |
| `--positions` | `experiments/public_models/public_edit_sweep.py` | フラグ無しなら従来と同一 |

## 締切に向けて残る最大のリスク

本文は**6ページ**で、ICASSP の 4+1 規定（技術内容4ページ＋参考文献のみ1ページ）を
超えている。これは追加実験以前からの状態であり、削減は内容判断なので著者に委ねている。
追加実験の結果は補足（17ページ）に置き、本文への追加は §4.4 の3文と §4.5 の2文に
とどめた。
