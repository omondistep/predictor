#!/bin/bash
cd /home/stdk/predictor
.venv/bin/python3 -c "from calibration_learner import retrain_from_results; retrain_from_results(force=True)"
