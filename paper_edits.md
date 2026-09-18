# paper_edits.md — manuscript changes in the icassp27-revision branch

Baseline: commit `44d6147` (author WIP 2026-09-17). Line counts are source lines of
`paper/icassp2027.tex`; the rendered page count stayed at 4 pages of text + 1 page of
references throughout (checked with `pdftotext`: page 4 ends with the Conclusion, page 5
begins with "7. REFERENCES").

## Main text

| # | location | before | after | Δ lines |
|---|---|---|---|---|
| 1 | Sec. 1, paragraph 2 | 7 sentences introducing probing and steering separately, each with a "studies on music generation models have also…" sentence | 3 sentences: one defining internal activations, one on probing (with the music citations folded in), one on steering (likewise) | −4 |
| 2 | Sec. 2.2, after Eq. (3) | — | "Eq. (3) combines linear concept erasure [INLP, LEACE] with mean patching [Wang et al.; Zhang & Nanda]." (T5; 3 new bib entries) | +1 |
| 3 | Sec. 3.3 "Evaluation prompts" | "For the public-model Bach evaluation in Fig. 4, probes and target-key means are estimated from 220 chorales, and the other 80 chorales are used for evaluation. The additional … cells in Table 1 use 20 search prompts …, 60 held-out final prompts, and 12 major target keys." | "For the deduplicated public-model Bach evaluation (Table 1a), probes and target-key means are estimated from 220 chorales; of the remaining 80, the first 20 by chorale number select the layer and the other 60 are the final prompts, with 12 major target keys. The additional cells in Table 1b,c use 20 search and 60 held-out prompts, with target means estimated from all pieces of the corpus." | 0 |
| 4 | Sec. 4.2, end of first paragraph | — | Three sentences: six-model range (T1), estimator with no intervention (T2), rank-23 (T4) | +3 |
| 5 | Sec. 4.5, paragraph | Fig. 4 references (left/right panels) | Table 1a references; one added sentence on SR_pc (T3) including the residual variant (REMI+ 0.567 kept, AMT-12L 0.106) | +2 |
| 6 | Sec. 4.5, last sentence | "…summarized in Table 1." | "…for AMT model sizes and Pop are summarized in Table 1b,c." | 0 |
| 7 | Fig. 4 (public results bars) | figure environment, 8 lines | removed; its content is Table 1(a) | −8 |
| 8 | Table 1 | `\scriptsize`, 6 columns, panels "Bach AMT size comparison" and "Pop model comparison" | `\small` (9 pt), 7 columns (Model, M_probe, SR, SR_rand, δD, SR_pc, Keys), panel (a) Bach deduplicated (3 rows) + (b) AMT sizes + (c) Pop; caption rewritten | +5 |
| 9 | Limitations | "The main run is the run with the highest success rate among the three augmented seeds." | removed (the range is now in 4.2) | −1 |
| 10 | Limitations | "The row space of W_ℓ also contains one direction that leaves the probe softmax unchanged. The edit component along this direction was 9–24% … and we have not tested whether removing it changes generation." | removed (tested; result in 4.2 and Supp. §13) | −2 |

Net: −5 source lines; additions of new content 5 sentences plus one clause (items 2, 4, 5), within the 6–8 line budget.

Not changed: Abstract, Conclusion (candidate sentences below, for the authors to choose),
Fig. 1–3, Sec. 2.1, 3.1, 3.2, 4.1, 4.3, 4.4. `spconf.sty` and margins untouched; no font
below 9 pt (`\small` in a 10 pt article = 9 pt; `\scriptsize` is gone).

## Candidate sentences (authors' choice — none inserted)

Abstract, last sentence (replace "These results show that music Transformers represent replaceable key information that affects generation."):
- A. "These results show that music Transformers represent replaceable key information that affects generation, and that probe accuracy and edit margin should be reported together."
- B. "We therefore recommend reporting probe accuracy alongside the edit margin of the same subspace, since the first does not predict the second."
- C. "Music Transformers thus represent replaceable key information that affects generation; where a representation is used can only be established by editing it, so we suggest reporting edit margins next to probe accuracies."

Conclusion, last sentence (after "Reading information from an internal representation and using it during generation must therefore be measured separately."):
- A. "We suggest that studies of music representations report probe accuracy and edit margin side by side."
- B. "A practical consequence is that a probe result should be accompanied by the edit margin of the same subspace before it is read as evidence of use."
- C. "Reporting both measures for the same subspace, as Table 1 does, is the minimal standard we propose for future work."

Space: page 4 currently has about 10 free lines, so one Abstract and one Conclusion sentence fit without further cuts.

## Supplement (paper/icassp2027_supp.tex), +200/−? lines, 20 pages

- §1 and §7: qualifier that the Table 1(b, c) targets were estimated with the prompt chorales included; the deduplicated evaluation (§29) is the exception.
- §6: new paragraph, the estimator on unedited continuations (T2).
- §13: new paragraph, rank-23 (T4).
- §14: new paragraph, the pitch-class control on the three public checkpoints (T3), with its non-uniform outcome.
- §23: new Table S13b and text, all six models at their re-selected layers (T1); the layer-4 transfer table stays as the transfer condition.
- §26: pointer to the deduplicated δD values.
- §27: Fig. S9c caption relabelled as exploratory (raw key-hit, K1-norm, 80 pooled prompts).
- §29 (new): "Deduplicated Bach evaluation (220/20/60)", rule, layer selection, Table S20, comparison with the pooled scan.
- §32: first weakening finding now reports both the transfer condition and the re-selected six runs.
- Index table: four rows added.

## Figures regenerated from artifacts

- `fig2_main_results_bars.pdf`: unchanged values; script now reads `verdict.json` instead of literals.
- `fig4_public_results_bars.pdf`: regenerated from the dedup cells (kept in `results/figures/` for the supplement or a talk; no longer in the main text).
- `fig2_combined_results.pdf`: 2×2 variant (ours + public, SR + δD) available if the authors prefer a figure over Table 1(a).
