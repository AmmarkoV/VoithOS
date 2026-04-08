#!/bin/bash
# setup.sh — one-shot setup for VoithOS and all its tools.
#
# Creates a single venv at the project root and installs every dependency
# needed by:
#   • perception_service.py  (webcam, mic, VLM client, TTS)
#   • client.py / video_qa.py (Gradio VLM client)
#   • microphone.py           (Vosk STT)
#   • greekTTS.py / webcam.py (Kokoro TTS)
#   • Y-MAP-Net               (TensorFlow pose / segmentation / depth)
#
# Y-MAP-Net is cloned from GitHub if the directory is missing.
# A symlink Y-MAP-Net/venv -> ../venv lets runYMAPNet.sh find the venv
# without modification.
#
# Vosk models (English + Greek) are downloaded to ~/.cache/vosk once.

set -euo pipefail

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"
ROOT="$(pwd)"   # canonical absolute path

echo "============================================================"
echo " VoithOS setup"
echo " Root: $ROOT"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[ 1/6 ] Checking system packages …"

SYSTEM_DEPS=(
    python3-venv
    python3-pip
    git
    wget
    unzip
    zip
    curl
    ffmpeg
    espeak-ng
    libportaudio2
    portaudio19-dev
    libhdf5-dev
    libsndfile1
    libgl1
)

MISSING=()
for pkg in "${SYSTEM_DEPS[@]}"; do
    if ! dpkg-query -W --showformat='${Status}\n' "$pkg" 2>/dev/null \
            | grep -q "install ok installed"; then
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  Installing missing packages: ${MISSING[*]}"
    sudo apt-get install -y "${MISSING[@]}"
else
    echo "  All system packages present."
fi

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "[ 2/6 ] Setting up Python venv …"

if [ ! -d venv ]; then
    echo "  Creating venv …"
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
echo "  Active Python: $(python3 --version)  $(which python3)"

# ── 3. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[ 3/6 ] Installing Python packages …"

# Upgrade pip/setuptools quietly
pip install -q --upgrade pip setuptools wheel

# Core perception / audio
pip install \
    vosk \
    sounddevice \
    opencv-python \
    numpy

# Gradio VLM client
pip install \
    gradio \
    gradio_client

# Kokoro TTS (supports Greek via espeak-ng)
pip install \
    kokoro \
    soundfile

# Argos translate (offline en↔el translation)
pip install argostranslate

# Y-MAP-Net: TensorFlow with optional CUDA support + all model deps
pip install \
    "tensorflow[and-cuda]" \
    tensorflow-model-optimization \
    tf_keras \
    numba \
    tensorboard \
    tensorboard-plugin-profile \
    etils \
    importlib_resources \
    wget \
    tf2onnx \
    onnx \
    onnxruntime

echo "  Python packages installed."

# ── 4. Vosk language models ───────────────────────────────────────────────────
echo ""
echo "[ 4/6 ] Checking Vosk language models …"

VOSK_CACHE="$HOME/.cache/vosk"
mkdir -p "$VOSK_CACHE"

download_vosk_model() {
    local name="$1"
    local url="$2"
    local zip="${name}.zip"
    if [ -d "$VOSK_CACHE/$name" ]; then
        echo "  ✓ $name already present"
    else
        echo "  Downloading $name …"
        wget -q --show-progress -O "/tmp/$zip" "$url"
        unzip -q "/tmp/$zip" -d "$VOSK_CACHE"
        rm "/tmp/$zip"
        echo "  ✓ $name installed to $VOSK_CACHE/$name"
    fi
}

# Small English model (~40 MB) — fast, good enough for commands
download_vosk_model \
    "vosk-model-small-en-us-0.15" \
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# Greek model (~1.5 GB) — only full model available
download_vosk_model \
    "vosk-model-el-gr-0.7" \
    "https://alphacephei.com/vosk/models/vosk-model-el-gr-0.7.zip"

# ── 5. Argos translate language pack ─────────────────────────────────────────
echo ""
echo "[ 5/6 ] Setting up Argos translate (en ↔ el) …"

if argospm list 2>/dev/null | grep -q "translate-en_el"; then
    echo "  ✓ en→el pack already installed"
else
    echo "  Installing language packs …"
    argospm update
    argospm install translate-en_el
    argospm install translate-el_en
fi

# ── 6. Y-MAP-Net ──────────────────────────────────────────────────────────────
echo ""
echo "[ 6/6 ] Setting up Y-MAP-Net …"

YMAPNET_DIR="$ROOT/Y-MAP-Net"

if [ ! -f "$YMAPNET_DIR/YMAPNet.py" ]; then
    echo "  Cloning Y-MAP-Net …"
    git clone https://github.com/FORTH-ICS-CVRL-HCCV/Y-MAP-Net "$YMAPNET_DIR"
else
    echo "  ✓ Y-MAP-Net already present"
    # Pull latest changes
    echo "  Pulling latest changes …"
    git -C "$YMAPNET_DIR" pull --ff-only || \
        echo "  (pull skipped — local changes or detached HEAD)"
fi

# Symlink Y-MAP-Net/venv -> ../venv so runYMAPNet.sh finds the venv
if [ ! -e "$YMAPNET_DIR/venv" ]; then
    echo "  Creating venv symlink: Y-MAP-Net/venv -> ../venv"
    ln -s ../venv "$YMAPNET_DIR/venv"
elif [ -L "$YMAPNET_DIR/venv" ]; then
    echo "  ✓ Y-MAP-Net/venv symlink already exists"
else
    echo "  ⚠ Y-MAP-Net/venv exists as a real directory — leaving it alone."
    echo "    If you want a shared venv, remove it and re-run this script."
fi

# Download the YMAPNet model weights if not already present
if [ ! -d "$YMAPNET_DIR/2d_pose_estimation" ]; then
    echo "  Downloading 2d_pose_estimation model weights …"
    cd "$YMAPNET_DIR"
    wget -q --show-progress http://ammar.gr/2d_pose_estimation.zip
    unzip -q 2d_pose_estimation.zip
    rm 2d_pose_estimation.zip
    cd "$ROOT"
    echo "  ✓ Model weights downloaded"
else
    echo "  ✓ 2d_pose_estimation weights already present"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo " Run the perception service:"
echo "   python3 perception_service.py --ymapnet-dir ./Y-MAP-Net"
echo ""
echo " Run Y-MAP-Net directly:"
echo "   cd Y-MAP-Net && ./runYMAPNet.sh"
echo "   (or: python3 Y-MAP-Net/runYMAPNet.py)"
echo "============================================================"
