#!/usr/bin/env python3
# Dependencies:
#   python3 -m pip install gradio_client opencv-python
#
# Example:
#   python3 video_qa_frames.py --input video.mp4 \
#     --question "Is a person touching a car with his left hand?" \
#     --question "Is a person touching a car with his right hand?" \
#     --every 5 \
#     -o per_frame_results.csv

import argparse
import csv
import os
import re
import sys
import time
import tempfile

import cv2
from gradio_client import Client, handle_file


def sanitize_string(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = s.replace("\\", "")
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    s = s.replace("\n", " ").replace("\r", "")
    s = s.replace('"', '\\"')
    return s


def extract_response_text(result) -> str:
    # Matches your earlier pattern; falls back to str()
    try:
        return result[0][0][1]
    except Exception:
        return str(result)


def safe_header_name(q: str, i: int) -> str:
    # Make readable CSV headers; keep it deterministic
    q_clean = re.sub(r"\s+", " ", q.strip())
    if len(q_clean) > 40:
        q_clean = q_clean[:40].rstrip() + "…"
    return f"q{i+1}: {q_clean}"


def main():
    ap = argparse.ArgumentParser(description="Per-frame multi-question video QA -> CSV (rows=frames, cols=questions).")
    ap.add_argument("--input", required=True, help="Path to input video (e.g., .mp4)")
    ap.add_argument("--question", action="append", default=[], help="Question/prompt (repeatable).")

    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", default="8080")

    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=100)
    ap.add_argument("--greek", action="store_true")

    # Frame sampling controls
    ap.add_argument("--every", type=int, default=1, help="Process every Nth frame (default: 1 = all frames).")
    ap.add_argument("--max_frames", type=int, default=0, help="Stop after this many processed frames (0 = no limit).")
    ap.add_argument("--start_frame", type=int, default=0, help="Start processing from this frame index.")

    # Context control:
    ap.add_argument(
        "--keep_context_within_frame",
        action="store_true",
        help="If set, do NOT reset_state between questions for the same frame."
             " (Faster, but questions can influence each other.)"
    )

    ap.add_argument("-o", "--output", default="video_qa_per_frame.csv")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.question:
        print("Error: provide at least one --question.", file=sys.stderr)
        sys.exit(1)

    if args.every < 1:
        print("Error: --every must be >= 1.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.input}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 0.0  # unknown

    client = Client(f"http://{args.ip}:{args.port}")

    # Prepare CSV headers: metadata + one col per question
    question_headers = [safe_header_name(q, i) for i, q in enumerate(args.question)]
    headers = ["video_path", "frame_index", "timestamp_sec"] + question_headers

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    frame_idx = 0

    # Seek to start_frame if requested
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        frame_idx = args.start_frame

    # Temp directory for frame JPEGs (Gradio handle_file wants a filepath)
    with tempfile.TemporaryDirectory(prefix="video_qa_frames_") as tmpdir, \
         open(args.output, "w", newline="", encoding="utf-8") as f:

        w = csv.writer(f)
        w.writerow(headers)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Decide whether to process this frame
            if (frame_idx - args.start_frame) % args.every != 0:
                frame_idx += 1
                continue

            # Compute timestamp
            timestamp_sec = (frame_idx / fps) if fps and fps > 0 else ""

            # Write frame image to disk
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:09d}.jpg")
            # JPEG encode; if it fails, skip
            if not cv2.imwrite(frame_path, frame):
                frame_idx += 1
                continue

            # Upload this frame once (handle_file wraps path for the client)
            frame_file = handle_file(frame_path)

            row = [args.input, frame_idx, timestamp_sec]

            # For each question, query the server
            for qi, q in enumerate(args.question):
                try:
                    if not args.keep_context_within_frame or qi == 0:
                        client.predict(api_name="/reset_state")

                    # NOTE: Keep this aligned with your server.
                    # If your app expects input_video=... or input_image=..., rename here.
                    client.predict(
                        input_images=[frame_file],
                        input_text=q,
                        api_name="/transfer_input"
                    )

                    result = client.predict(
                        chatbot=[],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_length_tokens=args.max_tokens,
                        repetition_penalty=1.1,
                        max_context_length_tokens=4096,
                        greek_translation=args.greek,
                        api_name="/predict"
                    )

                    ans = sanitize_string(extract_response_text(result))
                except Exception as e:
                    ans = f"ERROR: {sanitize_string(str(e))}"

                row.append(ans)

            w.writerow(row)
            f.flush()

            processed += 1
            print(f"\r   Processed frame {frame_idx} -> row {processed}            ",end="",flush=True)

            if args.max_frames and processed >= args.max_frames:
                break

            frame_idx += 1

    cap.release()
    print(f"Saved CSV to: {args.output}")

    os.system("python3 plot_video_qa_results.py --video %s --csv %s --save_video" % (args.input,args.output)) 



if __name__ == "__main__":
    main()

