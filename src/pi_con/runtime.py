"""Runtime utilities: device resolution, autograd helpers, parameter counting, JSON I/O."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _resolve_torch_device(device_preference: str) -> torch.device:
    """What: 解析使用者指定的裝置偏好並回傳可用裝置。"""
    preference = device_preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("指定 --device cuda，但目前環境沒有可用 CUDA。")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("指定 --device mps，但目前環境沒有可用 Metal (MPS)。")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"不支援的 device: {device_preference}")


def configure_torch_runtime(device_preference: str) -> torch.device:
    """What: 啟用 PyTorch 執行環境並回傳實際使用裝置。"""
    torch.set_float32_matmul_precision("high")
    device = _resolve_torch_device(device_preference)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """What: 用 autograd 計算一階偏導。

    Why: physics residual 依賴對 (x, y, t) 的偏導，若輸出與輸入無關則直接回零，
         避免在 sparse-data 主線上出現 silent None 梯度。
    """
    if y.grad_fn is None and not y.requires_grad:
        return torch.zeros_like(x)
    grad = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(x)
    return grad


def count_parameters(model: torch.nn.Module) -> int:
    """What: 計算可訓練參數總數。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_json(path: Path, data: dict) -> None:
    """What: 以格式化 JSON 寫出結構化輸出。"""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
