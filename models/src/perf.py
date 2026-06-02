"""Training-loop performance toggles, shared by training and bench.

Centralises the set of accelerators we can opt into: TF32 matmul,
BF16 autocast, cuDNN benchmark, torch.compile, DataLoader workers.
The bench evaluates each in isolation and in combination; once a
winning combo is identified, production training adopts the same
toggles via the same helpers.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


def apply_tf32(enabled: bool = True) -> None:
    """Enable / disable TF32 matmul on Ampere+ GPUs (L40s = Ada).

    Roughly 2x throughput ceiling vs FP32 on these GPUs with no
    measurable accuracy hit for the precision targets we care about.
    """
    mode = "high" if enabled else "highest"
    torch.set_float32_matmul_precision(mode)
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    logger.info("TF32 matmul: %s", enabled)


def apply_cudnn_benchmark(enabled: bool = True) -> None:
    """Toggle cuDNN benchmark mode (faster on stable shapes)."""
    torch.backends.cudnn.benchmark = enabled
    logger.info("cuDNN benchmark: %s", enabled)


@contextmanager
def bf16_autocast(enabled: bool = True) -> Iterator[None]:
    """BF16 autocast context.

    BF16 (not FP16) on Ampere+ GPUs: same dynamic range as FP32, no
    GradScaler bookkeeping. Typical 1.5-2x speedup. Caveat: rare PyG
    kernels can trip on bf16 (e.g., a few ViSNet paths). Smoke-test
    before adopting in production.
    """
    if enabled:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
        ):
            yield
    else:
        with torch.autocast(device_type="cuda", enabled=False):
            yield


def compile_model(
    model: torch.nn.Module,
    enabled: bool = True,
    dynamic: bool = True,
) -> torch.nn.Module:
    """Wrap a model in ``torch.compile`` if enabled.

    Args:
        model: nn.Module to wrap.
        enabled: If False, returns the model unchanged.
        dynamic: Pass through to torch.compile. PyG batches have
            varying total node counts; dynamic=True avoids
            recompilation per batch shape.

    Returns:
        Wrapped (or original) model.
    """
    if not enabled:
        return model
    logger.info("torch.compile: enabled (dynamic=%s)", dynamic)
    return torch.compile(model, dynamic=dynamic)


def dataloader_kwargs(
    num_workers: int = 0,
) -> dict:
    """Standard worker / pin_memory / persistent_workers settings.

    Args:
        num_workers: 0 = serial in-process loading; >0 = process pool
            with pin_memory and persistent_workers enabled.

    Returns:
        Dict suitable for ``DataLoader(**kwargs)``.
    """
    if num_workers <= 0:
        return {"num_workers": 0}
    return {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    }
