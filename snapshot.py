#!/usr/bin/env python3
"""
snapshot.py — grab a single frame from the SHM camera stream and save it.

Usage:
    python3 snapshot.py <tag>

Saves to:
    long_term_context/<tag>.jpg   — the captured frame
    long_term_context/<tag>.txt   — VLM description of the frame
"""

import json
import os
import sys
import time


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def describe_image(image_path: str, cfg: dict) -> str:
    from gradio_client import Client, handle_file

    vlm         = cfg.get("vlm", {})
    ip          = vlm.get("ip",          "127.0.0.1")
    port        = str(vlm.get("port",    "8080"))
    prompt      = vlm.get("prompt",      "Briefly and concisely describe what you see in this image.")
    temperature = vlm.get("temperature", 0.6)
    top_p       = vlm.get("top_p",       0.9)
    max_tokens  = vlm.get("max_tokens",  120)
    greek       = bool(vlm.get("greek",  False))

    client = Client(f"http://{ip}:{port}")
    client.predict(api_name="/reset_state")
    client.predict(
        input_images=[handle_file(image_path)],
        input_text=prompt,
        api_name="/transfer_input",
    )
    result = client.predict(
        chatbot=[],
        temperature=temperature,
        top_p=top_p,
        max_length_tokens=max_tokens,
        repetition_penalty=1.1,
        max_context_length_tokens=4096,
        greek_translation=greek,
        api_name="/predict",
    )
    return result[0][0][1].strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: snapshot.py <tag>", file=sys.stderr)
        sys.exit(1)

    tag = sys.argv[1]

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configuration.json")
    config      = load_config(config_path)

    shm_cfg     = config["shared_memory"]
    lib_dir     = shm_cfg["lib_dir"]
    descriptor  = shm_cfg["descriptor"]
    stream_name = shm_cfg["stream_name"]

    out_dir  = os.path.join(script_dir, "long_term_context")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{tag}.jpg")
    txt_path = os.path.join(out_dir, f"{tag}.txt")

    # ── capture frame from SHM ────────────────────────────────────────────────
    from shm_camera import SHMConsumer
    import cv2

    consumer = SHMConsumer(lib_dir, descriptor, stream_name)
    frame    = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frame = consumer.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        print("snapshot: could not read frame from SHM stream", file=sys.stderr)
        sys.exit(1)

    cv2.imwrite(img_path, frame)

    # ── describe via VLM ──────────────────────────────────────────────────────
    try:
        description = describe_image(img_path, config)
        with open(txt_path, "w") as f:
            f.write(description)
    except Exception as e:
        print(f"snapshot: VLM description failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(img_path)
    print(txt_path)


if __name__ == "__main__":
    main()
