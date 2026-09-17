"""Unit tests (stdlib unittest - no GPU, no network beyond localhost)."""
import csv
import os
import tempfile
import unittest

from servebench import bench_client, mock_server
from servebench.configs import ServingConfig, default_grid
from servebench.gpu_catalog import CATALOG, get
from servebench.serving_model import evaluate, fungibility_score, LLAMA_8B
from servebench.sweep import run as sweep_run


class TestConfigs(unittest.TestCase):
    def test_grid_size_and_uniqueness(self):
        grid = default_grid()
        self.assertEqual(len(grid), 72)
        self.assertEqual(len({c.short() for c in grid}), 72)

    def test_flags_map_to_vllm(self):
        f = ServingConfig().vllm_flags()
        for flag in ("--tensor-parallel-size", "--max-num-seqs",
                     "--max-num-batched-tokens", "--gpu-memory-utilization",
                     "--enable-chunked-prefill", "--enable-prefix-caching"):
            self.assertIn(flag, f)


class TestModel(unittest.TestCase):
    def test_deterministic(self):
        cfg = ServingConfig()
        g = get("H100-SXM")
        a, b = evaluate(cfg, g), evaluate(cfg, g)
        self.assertEqual(a.tok_per_s_gpu, b.tok_per_s_gpu)

    def test_all_skus_feasible_for_default(self):
        for g in CATALOG:
            m = evaluate(ServingConfig(), g)
            self.assertTrue(m.feasible, g.name)
            self.assertGreater(m.tok_per_s_gpu, 0)

    def test_h100_beats_4090_per_gpu(self):
        # Sanity: flagship datacenter SKU should outrank a consumer card
        # in the mock model. If this ever flips, the model changed.
        cfg = ServingConfig(max_num_seqs=128)
        h = evaluate(cfg, get("H100-SXM")).tok_per_s_gpu
        r = evaluate(cfg, get("RTX-4090")).tok_per_s_gpu
        self.assertGreater(h, r)

    def test_oversized_batch_thrashes_small_gpu(self):
        # The cross-SKU effect: max_num_seqs=256 tuned for 80GB thrashes
        # on a 24GB card, while 32 is fine there.
        big = ServingConfig(max_num_seqs=256)
        small = ServingConfig(max_num_seqs=32)
        g4090 = get("RTX-4090")
        self.assertLess(evaluate(big, g4090).tok_per_s_gpu,
                        evaluate(small, g4090).tok_per_s_gpu)
        # ...but the big batch is fine on the H100 (no thrash there)
        h100 = get("H100-SXM")
        self.assertGreaterEqual(evaluate(big, h100).tok_per_s_gpu,
                                evaluate(small, h100).tok_per_s_gpu * 0.9)

    def test_chunked_prefill_helps_mixed_load(self):
        g = get("A100-80GB")
        on = evaluate(ServingConfig(enable_chunked_prefill=True), g)
        off = evaluate(ServingConfig(enable_chunked_prefill=False), g)
        self.assertLess(on.itl_s, off.itl_s)

    def test_infeasible_when_weights_do_not_fit(self):
        tiny = ServingConfig(gpu_memory_utilization=0.01)
        m = evaluate(tiny, get("RTX-4090"))
        self.assertFalse(m.feasible)

    def test_fungibility_score_bounds(self):
        for cfg in default_grid()[:10]:
            s, per_sku = fungibility_score(cfg, CATALOG)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
            self.assertEqual(set(per_sku), {g.name for g in CATALOG})


class TestMockServerClient(unittest.TestCase):
    def test_e2e_measurement_path(self):
        # The client's timing path is validated against the mock server:
        # with TIME_SCALE=1.0 the measured TTFT should match the mock's
        # simulated TTFT within scheduling overhead.
        cfg = ServingConfig(max_num_seqs=64)
        old_scale = mock_server.TIME_SCALE
        mock_server.TIME_SCALE = 1.0
        try:
            srv = mock_server.run_server(cfg, get("L40S"))
            try:
                url = f"http://127.0.0.1:{srv.server_port}"
                prompt = "word " * 2000  # 2000 mock prompt tokens
                r = bench_client.benchmark(url, n_requests=4, concurrency=2,
                                           prompt=prompt, max_tokens=4)
                self.assertEqual(len(r.ok_stats), 4)
                m = evaluate(cfg, get("L40S"), LLAMA_8B,
                             workload=[(2000, 4, 1.0)])
                expected_ttft = m.ttft_s  # TIME_SCALE == 1.0
                for s in r.ok_stats:
                    self.assertEqual(s.n_tokens, 4)
                    # measured can't beat the sleeps; overhead bounded
                    self.assertGreaterEqual(s.ttft_s, expected_ttft * 0.9)
                    self.assertLess(s.ttft_s, expected_ttft + 0.6)
                self.assertGreater(r.agg_tok_per_s(), 0)
            finally:
                srv.shutdown()
        finally:
            mock_server.TIME_SCALE = old_scale


class TestSweep(unittest.TestCase):
    def test_sweep_writes_csv_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            sweep_run([get("H100-SXM"), get("RTX-4090")], d)
            csv_path = os.path.join(d, "results.csv")
            md_path = os.path.join(d, "REPORT.md")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(md_path))
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 144)  # 72 configs x 2 SKUs
            self.assertTrue(all(r["feasible"] == "True" for r in rows))
            with open(md_path) as f:
                md = f.read()
            self.assertIn("MOCK", md)
            self.assertIn("Fungibility", md)


if __name__ == "__main__":
    unittest.main()
