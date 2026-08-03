#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/.venv/bin/python3" -c "from calibration_learner import auto_retrain; auto_retrain(force=True)"
