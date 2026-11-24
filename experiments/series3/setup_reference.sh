#!/bin/bash
# New run-local environment. Never updates the validated series2 environment.
set -Eeuo pipefail
export PIP_NO_CACHE_DIR=1
export XDG_CACHE_HOME=/run-series/cache
test "$(uname -s)" = Linux
test -f /run-series/kit/reference_gate.py
ENV_DIR=/run-series/reference-env
if test -e "$ENV_DIR"; then
  echo 'Refusing to reuse a partially built reference environment.' >&2
  exit 1
fi
/work/env-cpu/bin/python -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  'mlx[cpu]==0.32.4' 'numpy==2.4.6' \
  'transformers @ git+https://github.com/huggingface/transformers.git@430050f6f71195b76004b5d61bbb446ec0508fb3' \
  'mlx-lm @ git+https://github.com/jundot/mlx-lm.git@0c4c99192230f1943dd468d297f0b8a64ae7b7c2'
"$ENV_DIR/bin/python" -m pip freeze > /run-series/reference-environment.txt
DEADLINE_UNIX=$("$ENV_DIR/bin/python" -c 'import time; print(time.time()+900)')
timeout --signal=TERM --kill-after=30s 930s \
  "$ENV_DIR/bin/python" -u /run-series/kit/reference_gate.py \
  --output /run-series/REFERENCE_GATE.json --deadline-unix "$DEADLINE_UNIX"
