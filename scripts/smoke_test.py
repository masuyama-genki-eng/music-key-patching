"""Exercise data, training, checkpoint loading, capture and editing on a tiny model.

Uses temporary files and does not produce paper results or append a run ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Keep this installation check small and reproducible on GPU hosts as well.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import yaml

from experiments.data_and_models.generate_corpus import generate_split, write_split
from src.intervene.edit import SubspaceEditor
from src.model.train import TrainRun, train_one
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(1)
    cfg = yaml.safe_load((REPO / "configs/data_syn.yaml").read_text())
    with tempfile.TemporaryDirectory(prefix="tonal-smoke-") as temporary:
        root = Path(temporary)
        train_rows = generate_split("train", 8, cfg, workers=1)
        val_rows = generate_split("val", 4, cfg, workers=1)
        assert not {r["piece_seed"] for r in train_rows} & {
            r["piece_seed"] for r in val_rows
        }
        assert train_rows == generate_split("train", 8, cfg, workers=1)
        write_split(train_rows, root / "train.parquet")
        write_split(val_rows, root / "val.parquet")
        run = TrainRun(
            regime="R-Aug",
            seed=0,
            train_parquet=str(root / "train.parquet"),
            val_parquet=str(root / "val.parquet"),
            out_dir=str(root / "model"),
            model={"d": 32, "n_layers": 2, "n_heads": 2, "ctx": 512, "dropout": 0.0},
            max_steps=2,
            warmup_steps=1,
            batch_size=2,
            eval_every=1,
            eval_n_seqs=4,
            ckpt_every=1,
            ledger=False,
        )
        metrics = train_one(run)
        model = load_model(str(root / "model/final.pt"), "cpu").eval()
        ids = torch.tensor([train_rows[0]["token_ids"][:32]])
        with torch.no_grad():
            clean, acts = model(ids, capture=True)
            assert clean.shape == (1, 32, len(VOCAB))
            assert len(acts) == 2 and all(a.shape == (1, 32, 32) for a in acts)
            basis = torch.linalg.qr(torch.randn(32, 4)).Q
            sham = model(ids, editors={0: SubspaceEditor(basis, mode="sham")})
            torch.testing.assert_close(sham, clean, atol=2e-5, rtol=2e-5)
            editor = SubspaceEditor(basis, torch.ones(32), from_position=16)
            edited, changed = model(ids, capture=True, editors={0: editor})
            torch.testing.assert_close(
                changed[0][:, :16], acts[0][:, :16], atol=0, rtol=0
            )
            assert not torch.allclose(edited[:, 16:], clean[:, 16:])
            a = model.generate(ids, 4, rng=torch.Generator().manual_seed(7))
            b = model.generate(ids, 4, rng=torch.Generator().manual_seed(7))
            assert torch.equal(a, b)
        assert np.isfinite(metrics["final"]["val_loss"])
        print(
            json.dumps(
                {
                    "status": "passed",
                    "device": "cpu",
                    "train_pieces": 8,
                    "validation_pieces": 4,
                    "training_steps": 2,
                    "validation_loss": metrics["final"]["val_loss"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
