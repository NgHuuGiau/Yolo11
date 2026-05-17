from typing import Dict

import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device(requested_device: str = "auto") -> Dict[str, str]:
    requested = requested_device.lower().strip()
    cuda_ready = is_cuda_available()

    if requested == "cpu":
        return {
            "label": "CPU",
            "torch_device": "cpu",
            "yolo_device": "cpu",
            "gpu_name": None,
            "use_half": False,
        }

    if requested in {"auto", "cuda"} and cuda_ready:
        gpu_name = torch.cuda.get_device_name(0)
        return {
            "label": "CUDA",
            "torch_device": "cuda",
            "yolo_device": 0,
            "gpu_name": gpu_name,
            "use_half": True,
        }

    if requested == "cuda" and not cuda_ready:
        print("Warning: CUDA was requested but is not available. Falling back to CPU.")

    return {
        "label": "CPU",
        "torch_device": "cpu",
        "yolo_device": "cpu",
        "gpu_name": None,
        "use_half": False,
    }


def print_device_info(device_info: Dict[str, str]) -> None:
    print(f"Device selected: {device_info['label']}")
    if device_info["gpu_name"]:
        print(f"GPU: {device_info['gpu_name']}")

