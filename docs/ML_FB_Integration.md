# ML ↔ FB Model Integration

## Overview

The ML model (8-classifier ensemble) provides two things to the FB (Forebet-derived statistical) model:

1. **Feature importance weights** — tells the FB model which signals matter most
2. **Ensemble agreement + feature quality** — tells the FB model how confident to be

## ML Model Architecture

### Classifiers (4 per target, 8 total)

| Target | Classifiers |
|--------|------------|
| **1X2** | RandomForest, GradientBoosting, XGBoost, LightGBM |
| **O/U** | RandomForest, GradientBoosting, XGBoost, LightGBM |

All use probability calibration (Platt scaling) on chronological holdout.

### Feature Categories

| Type | Count | Examples |
|------|-------|---------|
| **Independent** | 113 | Form, goals, shots, injuries, H2H, common opponent analysis |
| **Market** | 15 | Odds, Forebet probabilities, derived market features |

Feature selection guarantees **≥30 independent features** out of 50 selected max, preventing the model from copying market odds.

## How ML Informs the FB Model

### 1. Feature Importance → Signal Weights

```
ML feature importances → _SIGNAL_MAP → FB signal weights
```

The `_SIGNAL_MAP` maps 128 ML features to 16 FB signal categories:

| FB Signal | ML Features | What It Adjusts |
|-----------|-------------|-----------------|
| `form` | 0-5, 100-105 | Form signal shift (×0.08 × weight) |
| `h2h` | 18-21 | Transitive analysis weight |
| `draw` | 23, 36, 6-8, 11, 116-119 | Draw detection thresholds |
| `goals` | 9-17 | Expected goals computation |
| `shots` | 46-51 | Shot-based adjustments |
| `injuries` | 67-84 | Injury impact weighting |
| `xg` | 90-92, 108-111 | Expected goals integration |

**Example**: If ML learns `form` features are very important (weight=1.0), the form signal shifts expected goals at full strength. If less important (weight=0.6), it's reduced by 40%.

### 2. Ensemble Agreement → Confidence

```
4 classifiers predict → measure variance → agreement score (0-1)
```

- **High agreement** (near 1.0): All classifiers agree → lower confidence thresholds → easier to reach "Near Certain"
- **Low agreement** (near 0.0): Classifiers disagree → raise confidence thresholds → harder to reach high confidence

Adjusts thresholds by ±1.5% max and final blend weight by ±4% max.

### 3. Feature Quality → Data Quality

```
Missing features + zero values → quality score (0.3-1.0)
```

Blended with base data quality (70/30):
```
data_quality = base_quality * 0.7 + ml_feature_quality * 0.3
```

Gates DC/DNB thresholds and probability dampening.

## Integration Points in `analyze_from_data()`

### Form Signal (line ~2498)
```python
_form_weight = ml_signal_weights.get("form", 1.0)
if derby["is_derby"]:
    _form_weight *= 0.6  # Derby reduces form impact
shift = fsig * 0.08 * _form_weight
```

### Transitive/H2H Signal (line ~2621)
```python
_h2h_weight = ml_signal_weights.get("h2h", 1.0)
trans_weight *= _h2h_weight
```

### Draw Tendency (line ~2898)
```python
_draw_weight = ml_signal_weights.get("draw", 1.0)
_draw_prob_adj = 0.28 - (_draw_weight - 1.0) * 0.05
_draw_margin_adj = 0.12 + (_draw_weight - 1.0) * 0.02
```

### Confidence Thresholds (line ~2803)
```python
_agree_adj = (ml_ensemble_agreement - 0.5) * 0.03
nc_thresh -= _agree_adj  # Near Certain threshold
hi_thresh -= _agree_adj  # High threshold
```

### Final Blend (line ~2682)
```python
_agree_boost = (ml_ensemble_agreement - 0.5) * 0.08
_w = min(0.65, signal_blend + _agree_boost)
p_home = p_home * (1 - _w) + _ph * _w
```

## Data Flow

```
Training Phase:
  Historical data → extract_features() → select_features()
  → Train 4 classifiers → Calibrate → Save model

Prediction Phase:
  Forebet scrape + odds + injuries
  → Load ML model
  → Get signal weights (feature importances → _SIGNAL_MAP)
  → Get ensemble agreement (classifier variance)
  → Get feature quality (missing data count)
  → Compute base probabilities (Poisson + ML + Forebet blend)
  → Apply ML-weighted signals (form, H2H, draw)
  → Blend signal-adjusted probs with base probs
  → Apply calibration corrections
  → Determine confidence (thresholds ± agreement)
  → Output picks (1X2, DC, DNB, O/U, BTTS)
```

## Key Design Principles

1. **ML improves FB, not replaces it** — ML weights adjust FB signal strengths, not override predictions
2. **Independent features dominate** — 113/128 features are independent of market odds
3. **Ensemble disagreement = caution** — When classifiers disagree, confidence is reduced
4. **Data quality gates risk** — Poor data quality raises DC/DNB thresholds
5. **Derby awareness** — Form signal reduced by 40% in local derbies
6. **Common opponent analysis** — ML learns optimal weighting of shared-opponent scoring/defense metrics
