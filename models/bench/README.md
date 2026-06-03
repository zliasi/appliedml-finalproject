# bench

`bench-inference.py` benchmarks the real warm-start cost per trained checkpoint: cold load (torch.load + build_model + load_state_dict, exactly what the application does), warm single-slab CPU inference latency, parameter count, and the checkpoint's recorded val MAE. Point it at a run's checkpoints directory. With no checkpoints yet, it falls back to building each backend from its config with random weights, which only smoke-tests that the backends build and run.

Run: `python bench/bench-inference.py --checkpoints runs/magmom-magmom21-v1p1/checkpoints` (writes `inference-results.json`).
