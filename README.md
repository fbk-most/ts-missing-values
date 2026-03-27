# ts-missing-values

A Python library for detecting and filling missing values in time series data.

It provides a complete pipeline — from preprocessing and gap detection to sporadic missing values interpolation and KNN-based gaps of clustered of missing values filling — built on top of [Darts](https://github.com/unit8co/darts).

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

# Given that all_series is the list of all TimeSeries
# Assume `main_series` is a TimeSeries with gaps and
# `candidates` is a list of related TimeSeries.
index = 4
main_sereis = all_series[index]
candidates = all_series.copy()
candidates.remove(main_series)

filled = universal_filling(
    series=main_series,
    candidates=candidates,
    # sporadic missing values filling
    num_values=3,
    distance=168,
    # gap filling
    k=3,
    metric="mae",
    transform="none",
)
```
## Examples
### `synthetic_traffic_gap_filling.ipynb`

Loads a synthetic traffic sensor dataset, then fills all missing values of a selected sensor using candidates series.<br>
Other than fillng the whole series, shows both sporadic and clustered missing values filling.

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
| `gap_transform` | Merge dense-`NaN` regions into contiguous gaps of clustered of missing values. |
| `sporadic_filling` | Fill sporadic missing values via neighbour interpolation and mean-seasonality fallback. |
| `top_k_similar_series` | Rank candidates by similarity to a reference series. |
| `gap_filling` | Fill gaps of clustered of missing values with a distance-weighted blend of the *k* most similar series. |
| `universal_filling` | End-to-end pipeline: gap transform → sporadic fill → gap fill → cleanup. |

### Comparison (`comparison`)

| Function | Description |
|---|---|
| `compare_series` | Compute a distance between two series (supports multiple metrics and transforms). |
| `percentile_transform` | Bin values into percentile buckets (shape-preserving). |
| `median_depth_transform` | Map values to their signed distance from the median. |
| `mean_transform` | Element-wise mean across a list of series. |

Built-in **metrics**: `mean absolutte distance (mae)`, `mean absolute percentage distance (mape)`, `symmetric mean absolute percentage distance (mape_symmetric)`, `root mean sqare error (rmse)`, `maximum absolute distance (max_distance)`, `pearson dissimilarity (pearson_dissimilarity)`, `cosine dissimilarity (cosine_dissimilarity)`.

Built-in **transforms**: `none`, `logarithmic (log)`, `distance from the median (depth)`, `percentile binning (percentile)`, `differencing (diff)`.

### Utility (`utility`)

| Function | Description |
|---|---|
| `extend_series` | Pad a series with `NaN` to reach new start/end timestamps. |
| `overlap_series` | Slice two series to their common time range. |
| `overlap_series_strict` | Keep only timestamps where both series have non-`NaN` values. |
| `get_extended_series` | Align a list of series to the same time range. |
| `align_to_main_series` | Extend/crop series to match a reference series. |
| `get_mean_series` | Element-wise mean across aligned series, ignoring NaNs. |

## License

This project is licensed under the [Apache License 2.0](LICENSE).
