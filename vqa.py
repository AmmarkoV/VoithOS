#!/usr/bin/env python3
"""
vqa.py — Visual Question Answering on the live camera frame.

Grabs one frame from the SHM camera stream, sends it to the VLM with the given
question, then forwards the result to ./command.sh as:
    <vqa><question>...</question><answer>...</answer></vqa>

Usage:
    python3 vqa.py "Is there anyone in the room?"
    python3 vqa.py What colour is the wall?
    echo "What do you see?" | python3 vqa.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
from gradio_client import Client, handle_file

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configuration.json")


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def grab_frame(args) -> "cv2.typing.MatLike":
    sys.path.insert(0, args.lib_dir)
    from shm_camera import SHMConsumer

    consumer = SHMConsumer(args.lib_dir, args.descriptor, args.stream)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frame = consumer.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.1)
    print("vqa: timed out waiting for camera frame", file=sys.stderr)
    sys.exit(1)


def ask_vlm(image_path: str, question: str, args) -> str:
    client = Client(f"http://{args.ip}:{args.port}")
    client.predict(api_name="/reset_state")
    client.predict(
        input_images=[handle_file(image_path)],
        input_text=question,
        api_name="/transfer_input",
    )
    result = client.predict(
        chatbot=[],
        temperature=args.temperature,
        top_p=args.top_p,
        max_length_tokens=args.max_tokens,
        repetition_penalty=1.1,
        max_context_length_tokens=4096,
        greek_translation=args.greek,
        api_name="/predict",
    )
    try:
        return result[0][0][1].strip()
    except Exception:
        return str(result).strip()


def main():
    cfg = load_config()
    vlm = cfg.get("vlm", {})
    shm = cfg.get("shared_memory", {})

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="Question to ask (or read from stdin)")
    ap.add_argument("--ip",          default=vlm.get("ip",          "127.0.0.1"))
    ap.add_argument("--port",        default=str(vlm.get("port",    "8080")))
    ap.add_argument("--temperature", type=float, default=vlm.get("temperature", 0.6))
    ap.add_argument("--top_p",       type=float, default=vlm.get("top_p",       0.9))
    ap.add_argument("--max_tokens",  type=int,   default=vlm.get("max_tokens",  120))
    ap.add_argument("--greek",       action="store_true", default=bool(vlm.get("greek", False)))
    ap.add_argument("--lib-dir",     default=shm.get("lib_dir",     ""))
    ap.add_argument("--descriptor",  default=shm.get("descriptor",  "voithos_video.shm"))
    ap.add_argument("--stream",      default=shm.get("stream_name", "voithos_cam"))
    args = ap.parse_args()

    question = " ".join(args.question) if args.question else sys.stdin.read()
    question = question.strip()
    if not question:
        sys.exit(0)

    if not args.lib_dir:
        print("vqa: shared_memory.lib_dir not set in configuration.json", file=sys.stderr)
        sys.exit(1)

    frame = grab_frame(args)

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, frame)
        answer = ask_vlm(tmp_path, question, args)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    payload = f"<vqa><question>{question}</question><answer>{answer}</answer></vqa>"
    print(payload)

    subprocess.Popen(
        [os.path.join(SCRIPT_DIR, "command.sh"), payload],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
