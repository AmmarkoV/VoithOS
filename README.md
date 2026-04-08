# VoithOS

An experimental AI agent assistant that perceives its environment through a webcam and microphone, reasons about what it sees and hears using a Vision-Language Model (VLM), and can speak back in English or Greek.

The system is designed to run as a low-resource background service. All observations are written as files to a RAM-backed directory so that an agent can read them at any time.

---

## Architecture

```
camera_server.py          — sole owner of /dev/video0
       │ frames via shared memory (RAM, no disk I/O)
       ├── vision_loop           — periodic VLM query → vision_latest.txt
       └── scene_writer.py       — continuous YMAPNet inference → scene_latest.json

microphone.py / mic_loop  — Vosk offline STT → speech_latest.txt

All outputs → /dev/shm/voithos_context/  (tmpfs, RAM only)
```

The camera is opened **once** by `camera_server.py` and its frames are published to a POSIX shared memory buffer (`/dev/shm/voithos_video.shm`). All other processes subscribe as read-only consumers — no camera contention.

---

## Components

| File | Purpose |
|---|---|
| `perception_service.py` | Main daemon — spawns and supervises all sensors |
| `camera_server.py` | Captures webcam frames, publishes to shared memory |
| `shm_camera.py` | Clean `SHMProducer` / `SHMConsumer` wrappers around SharedMemoryVideoBuffers |
| `scene_writer.py` | Runs YMAPNet (pose, segmentation, depth) on SHM frames, writes `scene_latest.json` |
| `client.py` | Batch VLM client — query a Gradio VLM server over a directory of images |
| `video_qa.py` | Per-frame multi-question video QA, outputs CSV |
| `microphone.py` | Standalone Vosk STT script |
| `talk.py` / `talk.sh` | English TTS via Kokoro (`af_bella` voice) |
| `talkgreek.py` / `talkgreek.sh` | Translate English → Greek (Argos) then speak via Kokoro (`ef_dora` voice) |
| `snapshot.py` / `snapshot.sh` | Grab a single frame from SHM, describe it with the VLM, save to `long_term_context/` |
| `configuration.json` | Single config file for all parameters |
| `scripts/setup.sh` | One-shot installer for all dependencies |

### External dependencies (cloned/built by setup.sh)

| Dependency | Role |
|---|---|
| [Y-MAP-Net](https://github.com/FORTH-ICS-CVRL-HCCV/Y-MAP-Net) | Local neural network for 2D pose, segmentation, depth maps |
| [SharedMemoryVideoBuffers](https://github.com/AmmarkoV/SharedMemoryVideoBuffers) | Zero-copy camera bus via POSIX shared memory |

---

## Context directory (agent-readable files)

All outputs land in `/dev/shm/voithos_context/` (RAM, tmpfs):

| File | Updated by | Content |
|---|---|---|
| `vision_latest.txt` | vision_loop | Latest VLM scene description |
| `speech_latest.txt` | mic_loop | Latest recognised speech utterance |
| `scene_latest.json` | scene_writer | Structured YMAPNet observation: keypoints, segmentation scores, depth, vocabulary description |
| `log.jsonl` | all sensors | Append-only event log (one JSON object per line) |
| `status.json` | heartbeat | Service config snapshot, updated every 30 s |

`long_term_context/` (on disk) stores named snapshots from `snapshot.sh`.

---

## Setup

```bash
bash scripts/setup.sh
source venv/bin/activate
```

The script installs all Python packages into a single venv, downloads Vosk language models, sets up Argos translate (en↔el), clones and builds Y-MAP-Net and SharedMemoryVideoBuffers, and patches `configuration.json` with the correct library path.

---

## Configuration

Edit `configuration.json` before running:

```json
{
    "vlm": {
        "ip":   "127.0.0.1",
        "port": "8080",
        "prompt": "Briefly and concisely describe what you see in this image.",
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 120,
        "greek": false
    },
    "vision":      { "interval_sec": 30, "camera": 0 },
    "microphone":  { "language": "en-us" },
    "ymapnet":     { "enabled": true, "interval_sec": 2.0, "eco": 5.0 },
    "output":      { "dir": "/dev/shm/voithos_context" },
    "shared_memory": {
        "lib_dir": "/path/to/SharedMemoryVideoBuffers/src/python",
        "descriptor": "voithos_video.shm",
        "stream_name": "voithos_cam"
    }
}
```

The VLM server is a separate Gradio-based endpoint (e.g. a local or remote deepseek-vl2 instance). All other inference (STT, pose, TTS, translation) runs fully offline.

---

## Running

```bash
# Full service (webcam + mic + YMAPNet local perception)
python3 perception_service.py

# Microphone only
python3 perception_service.py --no-vlm --no-ymapnet

# VLM + YMAPNet, no microphone
python3 perception_service.py --no-mic

# Force CPU inference for YMAPNet
python3 perception_service.py --cpu
```

### TTS scripts (for agent use)

```bash
./talk.sh "I can see a person sitting at a desk."
./talkgreek.sh "There is a person in the room."
./talkgreek.sh --greek "Υπάρχει ένα άτομο στο δωμάτιο."
echo "$(cat /dev/shm/voithos_context/vision_latest.txt)" | ./talk.sh
```

### Named snapshot

```bash
./snapshot.sh entrance        # saves long_term_context/entrance.jpg + .txt
./snapshot.sh before_meeting
```

### Y-MAP-Net standalone

```bash
cd Y-MAP-Net && ./runYMAPNet.sh
cd Y-MAP-Net && ./runYMAPNet.sh --headless --fast --eco 5
```

---

## Resource usage

- **Camera**: opened once; frame sharing via RAM (no repeated `/dev/video0` opens)
- **VLM**: network call every `vision_interval_sec` seconds (default 30 s); no local GPU needed
- **YMAPNet**: runs with `--fast` + `--eco` flags (skips inference on static frames)
- **Vosk STT**: interrupt-driven audio stream, offline, ~40 MB model for English
- **Context files**: written to `/dev/shm/` (RAM), zero disk wear
