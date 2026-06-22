# Driving Risk Dataset

A driving behavior dataset collected from public web sources, comprising 15,000 samples across 6 features and 3 risk levels for driving risk assessment research.

## Overview

| Item | Detail |
|------|--------|
| Samples | 15,000 |
| Features | 6 |
| Risk Levels | 3 (Low / Medium / High) |
| Format | CSV |

## Feature Description

| Feature | Unit | Description |
|---------|------|-------------|
| `speed` | km/h | Instantaneous vehicle speed |
| `acceleration` | m/s² | Longitudinal acceleration |
| `brake_force` | 0–1 | Normalized brake intensity |
| `steering_rate` | deg/s | Steering angle rate of change |
| `following_distance` | m | Distance to leading vehicle |
| `blink_frequency` | blinks/min | Driver eye blink rate |

## Labels

| Value | Risk Level |
|-------|------------|
| 0 | Low Risk |
| 1 | Medium Risk |
| 2 | High Risk |

## Class Distribution

| Risk Level | Count | Proportion |
|------------|-------|------------|
| Low (0) | ~6,000 | ~40% |
| Medium (1) | ~5,700 | ~38% |
| High (2) | ~3,300 | ~22% |

## Usage

```python
import pandas as pd

url = "https://raw.githubusercontent.com/testii342/driving-risk-dataset/main/driving_risk_dataset.csv"
df = pd.read_csv(url)

X = df.drop("risk_level", axis=1)
y = df["risk_level"]
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{driving-risk-dataset,
  author    = {Liu, Tingwei},
  title     = {Driving Risk Dataset},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/testii342/driving-risk-dataset}
}
```

## License

This dataset is released under the [MIT License](LICENSE).
