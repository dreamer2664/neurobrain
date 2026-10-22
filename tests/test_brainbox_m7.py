"""M7 tests: int8 weight-only quantization path."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOK = os.path.join(ROOT, "data", "tokenizer.json")
TINY_LM = {"n_layer": 1, "n_head": 2, "d_model": 32, "ctx": 64}


class TestQuantMechanics(unittest.TestCase):
    def test_quantize_dequantize_roundtrip_error_bounded(self):
        import torch

        from brainbox.quant import dequantize_state_dict, quantize_state_dict

        torch.manual_seed(0)
        sd = {"w": torch.randn(64, 48) * 0.5, "norm": torch.randn(48)}
        q = quantize_state_dict(sd)
        self.assertEqual(q["w"]["q"].dtype, torch.int8)
        back = dequantize_state_dict(q)
        err = (back["w"] - sd["w"]).abs().max().item()
        scale_max = q["w"]["scale"].max().item()
        self.assertLessEqual(err, scale_max)  # within one quantization step
        self.assertTrue(torch.equal(back["norm"], sd["norm"]))  # 1-D kept fp32

    def test_quantized_tiny_gpt_generates_close_logits(self):
        import torch

        from brainbox.checkpoints import make_stub
        from brainbox.host import BrainHost

        tmp = tempfile.mkdtemp(prefix="bb_m7_")
        try:
            ck = os.path.join(tmp, "lm-tiny")
            make_stub(ck, TOK, seed=3, config_override=TINY_LM)
            fp = BrainHost(checkpoint=ck, threads=2)
            fp.boot()
            hosts = [fp]
            from brainbox.quant import quantize_model

            qhost = BrainHost(checkpoint=ck, threads=2)
            qhost.boot()
            quantize_model(qhost.model)
            a = fp.score("def f", top_k=5)
            b = qhost.score("def f", top_k=5)
            # same top-1 token from near-identical weights (tiny model, seed fixed)
            self.assertEqual(a["token_ids"][0], b["token_ids"][0])
            out = qhost.generate("def f", max_tokens=5, seed=0)
            self.assertEqual(out["tokens"], 5)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestInt8Checkpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m7ck_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sidecar_write_load_and_size(self):
        from brainbox.checkpoints import load_checkpoint_dir, make_stub
        from brainbox.host import BrainHost
        from brainbox.quant import write_int8_sidecar

        ck = os.path.join(self.tmp, "lm-tiny")
        make_stub(ck, TOK, seed=4, config_override=TINY_LM)
        model, tok, info = load_checkpoint_dir(ck)
        path = write_int8_sidecar(ck, model)
        fp32_bytes = os.path.getsize(os.path.join(ck, "weights.pt"))
        int8_bytes = os.path.getsize(path)
        # 0.6 not 0.25: this 332 KB toy pays torch.save per-tensor overhead
        # on ~30 small tensors; the real 17 MB model measures ~0.27x (M7 table).
        self.assertLess(int8_bytes, fp32_bytes * 0.6,
                        f"int8 {int8_bytes} not < 0.6x fp32 {fp32_bytes}")
        with open(os.path.join(ck, "config.json")) as f:
            cfg = json.load(f)
        self.assertIn("weights_int8_sha256", cfg)

        host = BrainHost(checkpoint=ck, threads=2, quantize="int8")
        desc = host.boot()
        self.assertEqual(desc["quantized"], "int8")
        out = host.generate("x", max_tokens=5, seed=0)
        self.assertEqual(out["tokens"], 5)

    def test_bench_int8_flag(self):
        from brainbox.cli import main
        from brainbox.checkpoints import make_stub

        ck = os.path.join(self.tmp, "lm-tiny")
        make_stub(ck, TOK, seed=4, config_override=TINY_LM)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["bench", "--checkpoint", ck, "--tokens", "10",
                         "--int8", "--json"])
        self.assertEqual(code, 0)
        d = json.loads(buf.getvalue())
        self.assertEqual(d["quantized"], "int8")
        self.assertEqual(d["gen_tokens"], 10)
        self.assertLess(d["peak_rss_mb"], 1600)

    def test_grip_quantized_forward(self):
        import torch

        from brainbox.quant import quantize_model
        from neurobrain.gripnet import GripNet

        torch.manual_seed(2)
        model = GripNet()
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        t = torch.zeros(1, 48, dtype=torch.long)
        with torch.no_grad():
            s0, o0, a0 = model(x, t)
        quantize_model(model)
        with torch.no_grad():
            s1, o1, a1 = model(x, t)
        self.assertEqual(s1.shape, s0.shape)
        xy = model.click_xy(s1, o1, 128)[0].tolist()
        self.assertTrue(0 <= xy[0] < 128 and 0 <= xy[1] < 128)
        self.assertEqual(int(a1.argmax()), int(a0.argmax()))


if __name__ == "__main__":
    unittest.main()
