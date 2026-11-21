"""A5: mlx_vlm.train with a decision-weighted loss.

Same CLI as `python -m mlx_vlm.train`. Default (WEIGHT_MODE=hard): every completion token is
weighted by (1 - p)^HARD_GAMMA, the model's current error on it (gradient stopped, floor 0.05), and
the loss is normalised by the total weight. Tokens the model already predicts (boilerplate) fade
out; the tokens that decide the action — wherever they appear, Thought or Action — dominate.

Why: in UI-TARS outputs the action is decided by 1-2 tokens (first coordinate digit, first typed
word); with uniform weights they are a small share of the loss and stay at 2+ nats while the
average loss looks converged.

Implementation: two small patches on mlx-vlm 0.8 internals, looked up at call time.
  1. dataset: completion_mask (int) is multiplied by ACTION_WEIGHT from the last "Action" token on.
  2. loss: the weights are read when the completion mask is fetched and applied in cross-entropy.
"""

import os
import sys

import mlx.core as mx
import numpy as np

import mlx_vlm.trainer.vlm.cross_entropy.dataset as ds
import mlx_vlm.trainer.vlm.cross_entropy.runtime as rt
from mlx_vlm.trainer.losses.cross_entropy import token_cross_entropy

W = float(os.environ.get("ACTION_WEIGHT", "4"))
# MODE "hard" (default): per-token weight (1 - p)^GAMMA. MODE "action": fixed span weight W on the
# Action span. "action" is kept for reference only: with template thoughts the decision is already
# made inside the Thought ("Next, I'll click "30 min"."), so up-weighting the Action span misses it.
MODE = os.environ.get("WEIGHT_MODE", "hard")
GAMMA = float(os.environ.get("HARD_GAMMA", "1"))
FLOOR = float(os.environ.get("HARD_FLOOR", "0.05"))
_state: dict = {}
_orig_process = ds.VisionDataset.process


def _process(self, row):
    out = _orig_process(self, row)
    cm = out.get("completion_mask")
    if cm is None:
        return out
    if "action_id" not in _state:
        ids = self.processor.tokenizer.encode("Action", add_special_tokens=False)
        _state["action_id"] = ids[0]
    ids = np.array(out["input_ids"]).reshape(-1)
    m = np.array(cm).reshape(-1).astype(np.int32)
    pos = np.flatnonzero((ids == _state["action_id"]) & (m > 0))
    if pos.size and MODE == "action":
        m[pos[-1]:] = m[pos[-1]:] * int(W)
    out["completion_mask"] = mx.array(m.reshape(np.array(cm).shape))
    return out


_orig_cm = rt._completion_mask


def _completion_mask(batch, assistant_id):
    m = _orig_cm(batch, assistant_id)
    _state["w"] = m[:, 1:].astype(mx.float32)
    return m


_orig_ce = rt.cross_entropy


def _cross_entropy(logits, targets, mask=None, *, ignore_index=-100, vision_mask=None):
    w = _state.pop("w", None)
    if w is None or mask is None or w.shape != targets.shape:
        return _orig_ce(logits, targets, mask, ignore_index=ignore_index, vision_mask=vision_mask)
    maskb = mask.astype(mx.bool_)
    if ignore_index is not None:
        maskb = mx.logical_and(maskb, targets != ignore_index)
    safe = mx.where(maskb, targets, 0)
    tl = token_cross_entropy(logits, safe, ignore_index=None)
    if MODE == "hard":
        # weight = how wrong the model still is on this token (gradient stopped), floored.
        # Boilerplate the model already predicts gets ~0 weight; the 1-2 decision tokens dominate.
        p = mx.exp(-mx.stop_gradient(tl.astype(mx.float32)))
        w = mx.maximum((1.0 - p) ** GAMMA, FLOOR)
    ww = mx.where(maskb, w, 0.0)
    loss = (tl * ww).sum() / mx.maximum(ww.sum(), 1.0)
    n = maskb.sum()
    correct = mx.logical_and(mx.argmax(logits, axis=-1) == safe, maskb).sum()
    return loss, {
        "weight": n, "num_tokens": n,
        "num_vision_tokens": mx.array(0, dtype=mx.int32) if vision_mask is None else vision_mask.astype(mx.bool_).sum(),
        "token_accuracy": correct.astype(mx.float32) / mx.maximum(n.astype(mx.float32), 1.0),
        "nll": loss,
    }


ds.VisionDataset.process = _process
rt._completion_mask = _completion_mask
rt.cross_entropy = _cross_entropy

if __name__ == "__main__":
    from mlx_vlm.train import main
    main(sys.argv[1:])
