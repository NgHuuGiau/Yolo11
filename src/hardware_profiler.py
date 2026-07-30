import multiprocessing
import platform
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO


MODEL_TIERS = {
    "n": {
        "key": "n",
        "name": "YOLO11n",
        "path": "models/yolo11n.pt",
        "vram_mb": 1024,
        "tier": 1,
        "imgsz": 416,
        "label": "Nano - siêu nhẹ",
        "description": "Phù hợp CPU / iGPU / GPU < 2GB VRAM",
    },
    "s": {
        "key": "s",
        "name": "YOLO11s",
        "path": "models/yolo11s.pt",
        "vram_mb": 2048,
        "tier": 2,
        "imgsz": 512,
        "label": "Small - nhẹ",
        "description": "Phù hợp GPU 2-4GB VRAM (GTX 1650, RTX 2050)",
    },
    "m": {
        "key": "m",
        "name": "YOLO11m",
        "path": "models/yolo11m.pt",
        "vram_mb": 4096,
        "tier": 3,
        "imgsz": 640,
        "label": "Medium - trung bình",
        "description": "Phù hợp GPU 4-6GB VRAM (RTX 3050 Ti, RTX 2060)",
    },
    "l": {
        "key": "l",
        "name": "YOLO11l",
        "path": "models/yolo11l.pt",
        "vram_mb": 8192,
        "tier": 4,
        "imgsz": 640,
        "label": "Large - nặng",
        "description": "Phù hợp GPU 6-10GB VRAM (RTX 3060-3080)",
    },
    "x": {
        "key": "x",
        "name": "YOLO11x",
        "path": "models/yolo11x.pt",
        "vram_mb": 12288,
        "tier": 5,
        "imgsz": 640,
        "label": "X-Large - siêu nặng",
        "description": "Phù hợp GPU > 10GB VRAM (RTX 3090, 4090)",
    },
}

TIER_ORDER = ["n", "s", "m", "l", "x"]


def detect_hardware():
    info = {
        "platform": platform.platform(),
        "cpu_cores": multiprocessing.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "has_cuda": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_vram_mb": 0,
        "gpu_vram_gb": 0,
        "tier_key": "n",
        "tier_info": None,
    }

    if info["has_cuda"]:
        gpu_props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = gpu_props.name
        info["gpu_vram_mb"] = gpu_props.total_memory // (1024 * 1024)
        info["gpu_vram_gb"] = round(info["gpu_vram_mb"] / 1024, 1)

    return info


def recommend_tier(hardware_info):
    vram = hardware_info["gpu_vram_mb"]

    if vram >= 12000:
        tier_key = "x"
    elif vram >= 8000:
        tier_key = "l"
    elif vram >= 4000:
        tier_key = "m"
    elif vram >= 2000:
        tier_key = "s"
    else:
        tier_key = "n"

    hardware_info["tier_key"] = tier_key
    hardware_info["tier_info"] = MODEL_TIERS[tier_key]
    return tier_key


def get_safe_fallback(tier_key):
    idx = TIER_ORDER.index(tier_key)
    for fallback_key in reversed(TIER_ORDER[: idx + 1]):
        if MODEL_TIERS[fallback_key]["tier"] <= MODEL_TIERS[tier_key]["tier"]:
            return fallback_key
    return "n"


def ensure_models_downloaded(tier_keys=None):
    if tier_keys is None:
        tier_keys = TIER_ORDER
    missing = []
    for key in tier_keys:
        t = MODEL_TIERS[key]
        p = Path(t["path"])
        if not p.exists():
            missing.append(t)

    if not missing:
        print("All YOLO models are already downloaded.")
        return

    print(f"Downloading {len(missing)} YOLO model(s)...")
    for t in missing:
        p = Path(t["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        model_filename = Path(t["path"]).name
        print(f"  -> {t['name']} ({t['path']})...")
        model = YOLO(model_filename)
        model.save(str(p))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        temp_file = Path(model_filename)
        if temp_file.exists() and temp_file.resolve() != p.resolve():
            temp_file.unlink()
        print(f"  Done: {t['path']}")
    print("All models ready.")


BENCHMARK_IMSZ_OPTIONS = [416, 512, 640, 800]
BENCHMARK_WARMUP = 5
BENCHMARK_FRAMES = 30


def run_benchmark(device_info, tier_keys=None):
    import time
    import numpy as np
    import cv2

    if tier_keys is None:
        tier_keys = TIER_ORDER

    print("=== BENCHMARK ===")
    print(f"Device: {device_info['label']}")
    print("Measuring FPS for each model tier at different image sizes...\n")

    test_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    results = []

    for key in tier_keys:
        t = MODEL_TIERS[key]
        model_path = Path(t["path"])
        if not model_path.exists():
            print(f"  Skip {t['name']} - model not found")
            continue

        for imgsz in BENCHMARK_IMSZ_OPTIONS:
            if imgsz < t["imgsz"]:
                continue

            try:
                model = YOLO(str(model_path))
                for _ in range(BENCHMARK_WARMUP):
                    model.predict(test_frame, imgsz=imgsz, device=device_info["yolo_device"], verbose=False)

                start = time.perf_counter()
                for _ in range(BENCHMARK_FRAMES):
                    model.predict(test_frame, imgsz=imgsz, device=device_info["yolo_device"], verbose=False)
                elapsed = time.perf_counter() - start
                fps = BENCHMARK_FRAMES / elapsed

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                results.append({"key": key, "name": t["name"], "imgsz": imgsz, "fps": round(fps, 1)})
                print(f"  {t['name']:10s} imgsz={imgsz}  → {fps:5.1f} FPS")
            except Exception as e:
                print(f"  {t['name']:10s} imgsz={imgsz}  → ERROR: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print()
    if not results:
        print("No benchmark results.")
        return results

    results.sort(key=lambda r: -r["fps"])
    print(f"{'Model':<12} {'imgsz':<8} {'FPS':<8}")
    print("-" * 30)
    for r in results:
        print(f"{r['name']:<12} {r['imgsz']:<8} {r['fps']:<8}")

    print("\nKhuyến nghị cho FPS ~15:")
    best_for_15 = None
    for r in sorted(results, key=lambda x: abs(x["fps"] - 15)):
        if best_for_15 is None:
            best_for_15 = r
        if abs(r["fps"] - 15) < abs(best_for_15["fps"] - 15):
            best_for_15 = r
    if best_for_15:
        print(f"  >>> {best_for_15['name']} imgsz={best_for_15['imgsz']} → {best_for_15['fps']} FPS")

    imgsz_options = sorted(set(r["imgsz"] for r in results))
    print(f"\n{'Model':<12}", end="")
    for imgsz in imgsz_options:
        print(f" {'imgsz=' + str(imgsz):<14}", end="")
    print()

    for key in tier_keys:
        print(f"{MODEL_TIERS[key]['name']:<12}", end="")
        for imgsz in imgsz_options:
            fps_val = next((r["fps"] for r in results if r["key"] == key and r["imgsz"] == imgsz), None)
            if fps_val:
                print(f" {fps_val:<14.1f}", end="")
            else:
                print(f" {'N/A':<14}", end="")
        print()

    print()
    return results


def print_hardware_report(hardware_info):
    print("=== HARDWARE REPORT ===")
    print(f"OS: {hardware_info['platform']}")
    print(f"CPU cores: {hardware_info['cpu_cores']}")
    print(f"RAM: {hardware_info['ram_gb']} GB")

    if hardware_info["gpu_name"]:
        print(f"GPU: {hardware_info['gpu_name']}")
        print(f"VRAM: {hardware_info['gpu_vram_gb']} GB ({hardware_info['gpu_vram_mb']} MB)")
    else:
        print("GPU: None (CPU only)")

    tier = hardware_info["tier_info"]
    print(f"\n=== RECOMMENDATION ===")
    print(f"Tier {tier['tier']} - {tier['label']}")
    print(f"Model: {tier['name']} ({tier['path']})")
    print(f"Image size: {tier['imgsz']} (cân bằng tốc độ - chất lượng)")
    print(f"VRAM yêu cầu: ~{tier['vram_mb']} MB")
    print()

    print("Bảng xếp hạng model theo cấu hình:")
    print(f"  {'Tier':<6} {'Model':<10} {'imgsz':<8} {'VRAM (MB)':<12} {'Phân loại':<25} {'Phù hợp'}")
    print(f"  {'-'*90}")

    all_tiers = [MODEL_TIERS[k] for k in TIER_ORDER]
    for t in all_tiers:
        marker = ">>>" if t["key"] == tier["key"] else "   "
        arrow = " <-- Phù hợp nhất" if t["key"] == tier["key"] else ""
        print(
            f"  {marker} {t['tier']:<2}  {t['name']:<10} {t['imgsz']:<6} {t['vram_mb']:<10} {t['label']:<30} {t['description']}{arrow}"
        )
    print()


def export_tensorrt(pt_path, imgsz=640, half=True):
    pt_path = Path(pt_path)
    engine_path = pt_path.with_suffix(".engine")
    if engine_path.exists():
        print(f"TensorRT engine already exists: {engine_path}")
        return str(engine_path)

    print(f"Exporting {pt_path.name} to TensorRT (this may take a few minutes)...")
    model = YOLO(str(pt_path))
    model.export(
        format="engine",
        half=half,
        imgsz=imgsz,
        device=0,
        verbose=False,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Done: {engine_path}")
    return str(engine_path)


def tensorrt_available() -> bool:
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def auto_optimize(model_path: str, imgsz: int, force: bool = False) -> str:
    if not tensorrt_available():
        return model_path
    import torch
    if not torch.cuda.is_available():
        return model_path
    from config import AUTO_TENSORRT, OPTIMIZE
    if not force and not AUTO_TENSORRT and OPTIMIZE != "tensorrt":
        return model_path
    engine_path = export_tensorrt(model_path, imgsz=imgsz)
    return engine_path if engine_path else model_path
