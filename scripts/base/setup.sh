#!/bin/bash
# Installs the Python dependencies required to run SEATS / baseline evaluation
# on Qwen-Omni with the bundled lmms-eval.
set -euo pipefail

python -m pip install --upgrade pip

pip install transformers==4.57.1
pip install loguru sacrebleu evaluate sqlitedict tenacity decord
pip install pytablewriter hf_transfer qwen_omni_utils
pip install ffmpeg moviepy nvitop
