"""Tests for the checkpoint packer (deliverable 7 + agent-01's weights_sha256)."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pack_checkpoint as pc  # noqa: E402
from neurobrain.model import GPT, GPTConfig  # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "checkpoints", "lm_baseline_step4600.pt")
TOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "tokenizer.json")


def test_pack_layout_and_hashes(tmp_path):
    if not os.path.exists(CKPT):
        return  # no baseline ckpt in this sandbox: covered on agent-04 boxes
    out = tmp_path / "lm-test"
    cfg = pc.pack(CKPT, str(out), TOK, name="lm-test",
                  metrics={"codebench pass@1 (greedy)": 0.0}, samples=[], notes="test pack")
    for f in ("config.json", "weights.pt", "tokenizer.json", "card.md"):
        assert (out / f).exists(), f
    on_disk = json.load(open(out / "config.json"))
    assert on_disk["kind"] == "lm"
    assert on_disk["weights_sha256"] == pc.sha256_file(str(out / "weights.pt"))
    assert on_disk["tokenizer_sha256"] == pc.sha256_file(TOK)
    assert on_disk["vocab_size"] == 2052 and on_disk["n_layer"] == 6
    card = (out / "card.md").read_text()
    assert "weights_sha256" in card and "codebench pass@1 (greedy)" in card
    # the packed weights reload into the same architecture
    sd = torch.load(out / "weights.pt", map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(vocab_size=on_disk["vocab_size"], n_layer=on_disk["n_layer"],
                          n_head=on_disk["n_head"], d_model=on_disk["d_model"],
                          ctx=on_disk["ctx"]))
    model.load_state_dict(sd)  # strict
