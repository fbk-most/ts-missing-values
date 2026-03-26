# ts-missing-values

A Python library for detecting and filling missing values in time series data.

It provides a complete pipeline — from preprocessing and gap detection to small-gap interpolation and KNN-based large-gap filling — built on top of [Darts](https://github.com/unit8co/darts).

## Installation

```bash
pip install git+https://github.com/fbk-most/ts-missing-values.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/fbk-most/ts-missing-values.git
```

### Requirements

- Python ≥ 3.12
- [Darts](https://github.com/unit8co/darts) ≥ 0.43.0

## Quick Start

```python
from darts import TimeSeries
from ts_missing_values import universal_filling, get_extended_series

# Assume `main_series` is a TimeSeries with gaps and
# `candidates` is a list of related TimeSeries.
candidates = get_extended_series([main_series] + candidates)

filled = universal_filling(
    main_series=main_series,
    candidate_series=candidates,
    period=24,          # seasonal period (e.g. 24 hours)
    top_k=3,            # number of similar series to blend
    metric="rmse",
    transform="none",
)
```

## API Overview

### Preprocessing (`preprocessing`)

| Function | Description |
|---|---|
| `eliminate_outliers` | Replace values beyond *mean ± Nσ* with `NaN`. |
| `preprocess_series` | Pipeline: outlier removal, interpolation, and optional log transform. |

### Gap Filling (`gap_filling`)

| Function | Description |
|---|---|
| `extract_gap_density` | Rolling-window density of `NaN` values. |
| `gap_transform` | Merge dense-`NaN` regions into contiguous large gaps. |
| `small_gap_filling` | Fill small gaps via neighbour interpolation and mean-seasonality fallback. |
| `top_k_similar_series` | Rank candidates by similarity to a reference series. |
| `large_gap_filling` | Fill large gaps with a distance-weighted blend of the *k* most similar series. |
| `universal_filling` | End-to-end pipeline: gap transform → small fill → large fill → cleanup. |

### Comparison (`comparison`)

| Function | Description |
|---|---|
| `compare_series` | Compute a distance between two series (supports multiple metrics and transforms). |
| `percentile_transform` | Bin values into percentile buckets (shape-preserving). |
| `median_depth_transform` | Map values to their signed distance from the median. |
| `mean_transform` | Element-wise mean across a list of series. |

Built-in **metrics**: `mae`, `mape`, `mape_symmetric`, `rmse`, `max_distance`, `correlation`, `cosine_similarity`.
Built-in **transforms**: `none`, `log`, `depth`, `percentile`, `diff`.

### Utility (`utility`)

| Function | Description |
|---|---|
| `extend_series` | Pad a series with `NaN` to reach new start/end timestamps. |
| `overlap_series` | Slice two series to their common time range. |
| `overlap_series_strict` | Keep only timestamps where both series have non-`NaN` values. |
| `get_extended_series` | Align a list of series to the same time range. |
| `align_to_main_series` | Extend/crop series to match a reference series. |
| `get_mean_series` | Element-wise `nanmean` across aligned series. |

## License

This project is licensed under the [Apache License 2.0](LICENSE).
