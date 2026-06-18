#!/usr/bin/env python3
"""Analyze the game/ dataset and show merge potential with predictor."""
import json
import sys
from pathlib import Path

GAME_DATA = Path("/home/stdk/game/data/historical_matches_combined.json")
PREDICT_PY = Path("/home/stdk/predictor/predict.py")

# Load game dataset
with open(GAME_DATA) as f:
    dataset = json.load(f)

print(f"Game dataset: {len(dataset)} historical match records")
print(f"Keys in record: {list(dataset[0].keys())}")

# Unique league codes
codes = sorted(set(r["league_code"] for r in dataset))
print(f"Unique league codes: {len(codes)}")

# Codes that start with each prefix
from collections import Counter
prefixes = Counter(sorted(c['league_code'][:2].lower() if len(c['league_code'])>=2 else c['league_code'].lower() for c in codes))
print(f"Code prefix distribution (first 20):")
for p, n in sorted(prefixes.items(), key=lambda x: -x[1])[:20]:
    print(f"  {p}: {n}")

# Extract detect_league from predict.py to test mappings
import re
with open(PREDICT_PY) as f:
    content = f.read()

fn_start = content.find("def detect_league")
fn_rest = content[fn_start:]
fn_end = re.search(r"\n\ndef ", fn_rest)
detect_code = fn_rest[:fn_end.start()] if fn_end else fn_rest

exec(compile(detect_code + "\ndetect_league_func = detect_league", "detect.py", "exec"))

mapped = {}
unmapped = []
for code in codes:
    key = detect_league_func(code)
    if key == "default":
        unmapped.append(code)
    mapped[code] = key

print(f"\nMapped to known profiles: {len(mapped) - len(unmapped)} / {len(codes)}")
print(f"Mapped to default: {len(unmapped)}")

if unmapped:
    print(f"Unmapped codes (first 30): {unmapped[:30]}")

# For unmapped codes, show country+league info to help mapping
print(f"\nSample unmapped codes with context:")
count = 0
seen = set()
for r in dataset:
    c = r["league_code"]
    if c in unmapped and c not in seen:
        print(f"  {c}: country={r.get('country','?')}, league={r.get('league_name','?')}")
        seen.add(c)
        count += 1
        if count >= 30:
            break
