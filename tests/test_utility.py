from darts import TimeSeries
import pandas as pd
import numpy as np
import pytest
from conftest import *

from ts_missing_values.utility import * 
from ts_missing_values.gap_filling import create_artificial_sporadic_missing_values, create_artificial_gap


ts_base = random_values_ts(time_index_base)
ts_empty = empty_values_ts(time_index_base)
ts_sporadic = create_artificial_sporadic_missing_values(ts_base, 20)
ts_gap = create_artificial_gap(ts_base)
ts_sporadic_and_gap = create_artificial_sporadic_missing_values(create_artificial_gap(ts_base), 20)
one_value_ts = random_values_ts(time_index_length_one)
ts_len_zero = TimeSeries.from_values([])

@pytest.mark.parametrize(['input', 'output'],[
    [ts_empty, True],
    [ts_len_zero, True],
    [ts_base, False],
    [ts_sporadic, False],
    [ts_gap, False],
    [ts_sporadic_and_gap, False]
])
def test_is_series_empty(input, output):
    assert is_series_empty(input) == output



ti_min = pd.date_range('03-01-2000 10:00:00', '20-03-2000 15:00:00', freq='min')
ti_min_with_gaps = ti_min[ti_min.day > 2]

@pytest.mark.parametrize(['input', 'output'],[
    [ts_base.time_index, 'h'],
    [ts_empty.time_index, 'h'],
    [ti_min, 'min'],
    [ti_min_with_gaps, 'min'],
    [one_value_ts, None],
    [pd.DatetimeIndex(["2020-01-01", pd.NaT]), None]
])
def test_extract_most_common_freq(input, output):
    assert extract_most_common_freq(input) == output

def test_return_timedelta_true_gives_timedelta():
    index = pd.date_range("2020-01-01", periods=5, freq="h")
    result = extract_most_common_freq(index, return_timedelta=True)
    assert result == pd.Timedelta(hours=1)

def test_return_timedelta_false_gives_offset():
    index = pd.date_range("2020-01-01", periods=5, freq="h")
    result = extract_most_common_freq(index, return_timedelta=False)
    assert result == to_offset("h")

#============================== time series =========================

import pandas as pd



ts1 = create_test_series([1, 2, 3], offset=0)
ts2 = create_test_series([3, 2, 6, 7, 9], offset=0)
ts3 = create_test_series([0, 0, 0, 0], offset=3)
ts4 = create_test_series([1, 2, 3], offset=3)
ts5 = create_test_series([np.nan, np.nan, np.nan], offset=0)
ts6 = create_test_series([1, np.nan, np.nan], offset=5)
ts7 = create_test_series([10, 11, 12], offset=2)

@pytest.mark.parametrize(['input', 'output_values'],[
    [[ts1, ts1], ts1.univariate_values()],
    [[ts1, ts2, ts3], [2, 2, 4.5, 3.5, 4.5, 0, 0]],
    [[ts1, ts4], [1,2,3,1,2,3]],
    [[ts1, ts5], [1,2,3]],
    # [[ts1, ts5], [1,2,3,np.nan,1,np.nan,np.nan,]], # what should the funcion do in this case? extend or keep only the real values?
])
def test_get_mean_series(input, output_values):
    assert (get_mean_series(input).univariate_values() == output_values).all()


@pytest.mark.parametrize(['main', 'other'],[
    [ts1, [ts1, ts2, ts3, ts4]],
    [ts1, [ts1, ts5]],
    [ts1, [ts1, ts6]],
])
def test_align_to_main_series(main, other):
    aligned_series = align_to_main_series(main, other)

    for ts in aligned_series:
        assert (ts.time_index == main.time_index).all()


def test_extend_series_new_start_is_none():
    sample_series = ts1
    new_end = sample_series.end_time() + 3 * sample_series.freq
    result = extend_series(sample_series, new_start=None, new_end=new_end)
    assert result.start_time() == sample_series.start_time()
    assert result.end_time() == new_end
 
 
def test_extend_series_new_end_is_none():
    sample_series = ts1
    new_start = sample_series.start_time() - 3 * sample_series.freq
    result = extend_series(sample_series, new_start=new_start, new_end=None)
    assert result.start_time() == new_start
    assert result.end_time() == sample_series.end_time()
 
 
def test_extend_series_both_none():
    sample_series = ts1
    result = extend_series(sample_series, new_start=None, new_end=None)
    assert result.start_time() == sample_series.start_time()
    assert result.end_time() == sample_series.end_time()


@pytest.mark.parametrize(['all_series'],[
    [[ts1, ts2, ts3]],
    [[ts1, ts6]]
])
def test_get_extended_series(all_series):
    all_series_extended = get_extended_series(all_series)

    starts = []
    ends = []
    lens = []
    for ts in all_series_extended:
        starts.append(ts.time_index[0])
        ends.append(ts.time_index[-1])
        lens.append(len(ts))

    assert len(set(starts)) == 1, f"Starts do not match: {starts}"
    assert len(set(ends)) == 1, f"Ends do not match: {ends}"
    assert len(set(lens)) == 1, f"Lengths do not match: {lens}"

def test_get_extended_series_different_indexes():
    tsA = TimeSeries.from_values([1,2,3])
    with pytest.raises(ValueError):
        get_extended_series([tsA, ts1])


def test_get_extended_series_no_date_index():
    tsA = TimeSeries.from_values([1,2,3])
    tsB = TimeSeries.from_values([4,5,6,2,3])
    get_extended_series([tsA, tsB])
    assert True


def test_overlap_series_strict():
    tsA = create_test_series([1, 2, 3, np.nan, 4], offset=0)
    tsB = create_test_series([            10, 20, 30], offset=3)

    tsA_res = create_test_series([4], offset=3)
    tsB_res = create_test_series([20], offset=3)
    tsA_out, tsB_out, stats_dict = overlap_series_strict(tsA, tsB, return_overlapping_stats=True)
    a, b = overlap_series_strict(tsA, tsB, return_overlapping_stats=False)

    assert tsA_res == tsA_out
    assert tsB_res == tsB_out

    assert tsA_res == a
    assert tsB_res == b

    assert stats_dict['num_values_used'] == 1
    assert stats_dict['num_values_not_used'] == 3
    assert 0.25 == pytest.approx(stats_dict['percentage_overlapped_vaues'])
    assert 1 == pytest.approx(stats_dict['percentage_gap_covered'])
    assert (1/6) == pytest.approx(stats_dict['percentage_symmetric_overlap'])

def test_overlap_series_strict2():
    tsA = TimeSeries.from_values([1, 2])
    tsB = TimeSeries.from_values([1]).shift(1)
    tsA_out, tsB_out, stats_dict = overlap_series_strict(tsA, tsB, return_overlapping_stats=True)
    assert True

def test_overlap_series_strict3():
    tsA = TimeSeries.from_values([1, np.nan, np.nan])
    tsB = TimeSeries.from_values([np.nan, np.nan, 1]).shift(1)
    tsA_out, tsB_out, stats_dict = overlap_series_strict(tsA, tsB, return_overlapping_stats=True)

    assert True

def test_overlap_series_strict_no_time_index():
    ts1 = TimeSeries.from_values([1,np.nan,3])
    ts2 = TimeSeries.from_values([1,1,1,1,1])
    ts1_res, ts2_res = overlap_series_strict(ts1, ts2)
    assert (ts1_res.univariate_values() == [1,3]).all()
    assert (ts2_res.univariate_values() == [1,1]).all()

def test_overlap_series_strict_not_overlappable():
    with pytest.raises(ValueError):
        overlap_series_strict(ts1, ts6)
    with pytest.raises(ValueError):
        overlap_series_strict(ts1, ts5)


def test_overlap_series():
    tsA = create_test_series([1, 2, 3, np.nan, 4], offset=0)
    tsB = create_test_series([            10, 20, 30], offset=3)

    tsA_res = create_test_series([np.nan, 4], offset=3)
    tsB_res = create_test_series([10, 20], offset=3)
    tsA_out, tsB_out = overlap_series(tsA, tsB)

    assert tsA_res == tsA_out
    assert tsB_res == tsB_out
    assert True

def test_overlap_series_not_overlappable():
    with pytest.raises(ValueError):
        overlap_series(ts1, ts6)


@pytest.mark.parametrize("periods,seasonality", [(1, 3), (2, 7), (10, 1)])
def test_generate_periodic_array(periods, seasonality):
    arr = generate_periodic_array(periods=periods, seasonality=seasonality, noise=False)
    assert arr.shape == (periods * seasonality,)

def test_generate_periodic_array_no_noise():
    result = generate_periodic_array(periods=3, seasonality=4, noise=False, plot=True)
    expected = np.array([0, 1, 2, 3] * 3, dtype=float)
    assert (result == expected).all()

def test_generate_periodic_array_with_noise():
    result = generate_periodic_array(periods=3, seasonality=4, noise=True)
    expected = np.array([0, 1, 2, 3] * 3, dtype=float)
    assert (result != expected).any()

def test_generate_periodic_array_zero_period():
    with pytest.raises(ValueError):
        generate_periodic_array(periods=0)


def test_generate_periodic_series_length_and_start():
    ts = generate_periodic_series(periods=3, seasonality=4, start=pd.Timestamp("2024-01-01"), freq="h", noise=False)
    assert len(ts) == 12
    assert ts.start_time() == pd.Timestamp("2024-01-01")

def test_generate_periodic_series_no_start():
    ts = generate_periodic_series(periods=3, seasonality=4, freq="h", noise=True, plot=True)
    assert len(ts) == 12
    assert ts.start_time() == 0
    assert ts.end_time() == 11
    

def test_generate_periodic_series_zero_period():
    with pytest.raises(ValueError):
        generate_periodic_series(periods=0, seasonality=4, start=pd.Timestamp("2024-01-01"), freq="h", noise=False, plot=True) 


from ts_missing_values.utility import _are_overlappable
def test_are_overlappable():
    assert _are_overlappable(ts1, ts2)
    assert not _are_overlappable(ts1, ts6)