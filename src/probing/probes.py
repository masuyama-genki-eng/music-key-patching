"""Linear (multinomial LR) and MLP probes with sequence-level splits (SPEC §3 A1).

Probes are trained on h_ℓ(t) -> κ(t). All splitting and bootstrap is at the
SEQUENCE level; metrics are token-level macro-F1 (24-class) with tonic-12 / mode-2
decompositions. Deterministic under seed.
"""
from __future__ import annotations
import dataclasses
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("probes")


# ------------------------------------------------------------------ splits
def split_by_sequence(seq_idx: np.ndarray, seed: int,
                      frac=(0.8, 0.1, 0.1)) -> dict[str, np.ndarray]:
    """Boolean masks over rows; every sequence lands wholly in one part."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq_idx)
    perm = rng.permutation(uniq)
    n1 = int(len(uniq) * frac[0]); n2 = int(len(uniq) * (frac[0] + frac[1]))
    parts = {"train": perm[:n1], "val": perm[n1:n2], "test": perm[n2:]}
    return {k: np.isin(seq_idx, v) for k, v in parts.items()}


# ------------------------------------------------------------------ metrics
def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 24) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def macro_f1_from_conf(m: np.ndarray) -> float:
    tp = np.diag(m).astype(float)
    prec = np.divide(tp, m.sum(0), out=np.zeros_like(tp), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros_like(tp), where=m.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp),
                   where=(prec + rec) > 0)
    present = m.sum(1) > 0                        # macro over classes present in y_true
    return float(f1[present].mean())


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1_24": macro_f1_from_conf(confusion(y_true, y_pred, 24)),
        "macro_f1_tonic12": macro_f1_from_conf(confusion(y_true % 12, y_pred % 12, 12)),
        "macro_f1_mode2": macro_f1_from_conf(confusion(y_true // 12, y_pred // 12, 2)),
        "acc": float((y_true == y_pred).mean()),
    }


def per_sequence_confusions(y_true, y_pred, seq_idx, k: int = 24):
    """(S, k, k) int64 + seq ids — bootstrap/jackknife unit for sequence-level CIs."""
    uniq, inv = np.unique(seq_idx, return_inverse=True)
    m = np.zeros((len(uniq), k, k), dtype=np.int64)
    np.add.at(m, (inv, y_true, y_pred), 1)
    return m, uniq


# ------------------------------------------------------------------ probes
@dataclasses.dataclass
class ProbeConfig:
    kind: str = "linear"            # "linear" | "mlp"
    hidden: int = 256               # mlp only
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 30
    batch_size: int = 8192
    patience: int = 5               # early stop on probe-val macro-F1
    seed: int = 0


def _build(kind: str, d: int, hidden: int, k: int = 24) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d, k)
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, k))


def train_probe(X: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray],
                cfg: ProbeConfig, device: str = "cuda") -> dict:
    """Returns {'report': test metrics, 'y_pred_test', 'weights' (linear only)}."""
    torch.manual_seed(cfg.seed)
    Xt = {k: torch.from_numpy(X[m]).float() for k, m in masks.items()}
    yt = {k: torch.from_numpy(y[m].astype(np.int64)) for k, m in masks.items()}
    mu, sd = Xt["train"].mean(0, keepdim=True), Xt["train"].std(0, keepdim=True) + 1e-6
    Xt = {k: (v - mu) / sd for k, v in Xt.items()}

    model = _build(cfg.kind, X.shape[1], cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = len(Xt["train"])
    g = torch.Generator().manual_seed(cfg.seed)

    def eval_f1(part: str) -> tuple[float, np.ndarray]:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xt[part]), 65536):
                preds.append(model(Xt[part][i:i + 65536].to(device)).argmax(-1).cpu())
        model.train()
        yp = torch.cat(preds).numpy()
        return macro_f1_from_conf(confusion(yt[part].numpy(), yp, 24)), yp

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = Xt["train"][idx].to(device), yt["train"][idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        f1, _ = eval_f1("val")
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_pred = eval_f1("test")
    out = {"report": metric_report(yt["test"].numpy(), y_pred),
           "val_macro_f1": best_f1, "y_pred_test": y_pred,
           "epochs_ran": epoch + 1}
    if cfg.kind == "linear":
        # probe was trained on (x - mu)/sd; fold the standardization back so that
        # weights/bias act on RAW activations: logits = W_raw x + b_raw
        lin = model if isinstance(model, nn.Linear) else model[0]
        W = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        W_raw = (W / sd.numpy()).astype(np.float32)
        b_raw = (b - W_raw @ mu.numpy().ravel()).astype(np.float32)
        out["weights"] = W_raw
        out["bias"] = b_raw
    return out
