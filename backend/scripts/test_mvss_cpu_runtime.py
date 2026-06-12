import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    backend_dir = Path(__file__).resolve().parents[1]
    mvss_dir = backend_dir / "MVSS-Net"

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    if str(mvss_dir) not in sys.path:
        sys.path.insert(0, str(mvss_dir))

    from models.mvssnet import get_mvss
    from common.tools import inference_single

    selected_device = torch.device("cpu")
    checkpoint_path = mvss_dir / "ckpt" / "mvssnet_casia.pt"

    print(f"Python executable: {sys.executable}")
    print(f"MVSS torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"selected device: {selected_device.type}")
    print("MVSS configured for CPU mode")

    started_at = time.perf_counter()
    model = get_mvss(
        backbone="resnet50",
        pretrained_base=True,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device
    )
    model.load_state_dict(
        checkpoint,
        strict=True
    )
    model.to(selected_device)
    model.eval()
    load_seconds = time.perf_counter() - started_at
    print(f"MVSS model loaded in {load_seconds:.3f} seconds")

    image = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )
    cv2.rectangle(
        image,
        (160, 160),
        (352, 352),
        (255, 255, 255),
        thickness=-1
    )

    started_at = time.perf_counter()

    with torch.inference_mode():
        mask, confidence = inference_single(
            img=image,
            model=model,
            th=0.5
        )

    inference_seconds = time.perf_counter() - started_at
    print(f"MVSS CPU inference completed in {inference_seconds:.3f} seconds")
    print(f"mask shape: {getattr(mask, 'shape', None)}")
    print(f"confidence: {float(confidence):.6f}")


if __name__ == "__main__":
    main()
