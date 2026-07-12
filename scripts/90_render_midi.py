"""Demo: render listenable MIDI pairs (clean vs key-edited continuation).

For a few test prompts: generate the clean continuation and a v_probe L4 edited
continuation (sustained subspace edit from the bar-9 boundary), and write standard
MIDI files (hand-rolled SMF-0 writer — no new dependencies) to results/samples/.
These are DEMO artifacts (ledgered as such), not SPEC metrics.

Usage: .venv/bin/python scripts/90_render_midi.py [--targets 7,4] [--n 2]
"""
from __future__ import annotations
import argparse
import logging
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import IVOCAB, VOCAB
from src.utils.ledger import append_entry

log = logging.getLogger("render_midi")

TICKS_16TH = 120                      # 480 PPQN, 16th grid
KEY_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def key_name(k: int) -> str:
    return KEY_NAMES[k % 12] + ("m" if k >= 12 else "")


from src.utils.notes import ids_to_notes as tokens_to_notes  # shared with figures


def _varint(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_midi(notes: list[tuple[int, int, int]], path: Path, bpm: int = 96) -> None:
    events = []                                   # (tick, order, bytes)
    for start, dur, pitch in notes:
        events.append((start * TICKS_16TH, 1, bytes([0x90, pitch, 80])))
        events.append(((start + dur) * TICKS_16TH, 0, bytes([0x80, pitch, 0])))
    events.sort(key=lambda e: (e[0], e[1]))
    track = bytearray()
    tempo = 60_000_000 // bpm
    track += b"\x00\xff\x51\x03" + struct.pack(">I", tempo)[1:]
    t = 0
    for tick, _, msg in events:
        track += _varint(tick - t) + msg
        t = tick
    track += b"\x00\xff\x2f\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--targets", default="7,4", help="target tonics (major), comma-sep")
    ap.add_argument("--n", type=int, default=2, help="prompts to render")
    ap.add_argument("--outdir", default=str(REPO / "results/samples"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    gen_cfg = yaml.safe_load((REPO / "configs/gen.yaml").read_text())
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    pw = np.load(REPO / "results/probing" / name / "probe_weights.npz")
    cm = np.load(REPO / "results/probing" / name / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"])
    mu = mu_targets_from_means(cm[f"layer_{args.layer}"])

    prompts = SW.select_prompts(str(REPO / "results/data_syn/test.parquet"), args.n)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    token_dump: dict = {"layer": args.layer, "model": name, "prompts": [], "conts": {}}

    def render(pi: int, suffix: str, cont: list[int]) -> None:
        full = prompts[pi].ids + cont
        p = outdir / f"prompt{pi}_{suffix}.mid"
        write_midi(tokens_to_notes(full), p)
        written.append(str(p.relative_to(REPO)))
        log.info("wrote %s (%d tokens)", p.name, len(full))

    clean = SW.generate_batch(model, prompts, lambda plen: None, gen_cfg, device, 8, 0)
    token_dump["prompts"] = [{"ids": p.ids, "src_key": p.src_key} for p in prompts]
    token_dump["conts"]["clean"] = clean
    for pi, p in enumerate(prompts):
        render(pi, f"src{key_name(p.src_key)}_clean", clean[pi])
    for tgt in [int(x) for x in args.targets.split(",")]:
        editor = SW.make_editor(V, mu[tgt], device)
        conts = SW.generate_batch(model, prompts, lambda plen: {args.layer: editor},
                                  gen_cfg, device, 8, 0)
        token_dump["conts"][f"edit_T{tgt}"] = conts
        for pi, p in enumerate(prompts):
            render(pi, f"src{key_name(p.src_key)}_edit{key_name(tgt)}_L{args.layer}", conts[pi])

    import json
    dump_path = outdir / "demo_tokens.json"
    dump_path.write_text(json.dumps(token_dump))
    written.append(str(dump_path.relative_to(REPO)))
    append_entry(stage="DEMO midi render", config=vars(args), seeds=[0],
                 artifacts=written,
                 note="listenable clean-vs-edited pairs + token dump; demo only, "
                      "not a SPEC metric")


if __name__ == "__main__":
    main()
