from darts import TimeSeries
import pandas as pd
import numpy as np
import pytest

from ts_missing_values.utility import is_series_empty
from ts_missing_values.gap_filling import create_artificial_sporadic_missing_values


time_index_base = pd.date_range('2000-01-03 10:00:00', '2000-03-21 15:00:00', freq='h')
time_index_overlapped = pd.date_range('2000-02-05 18:00:00', '2000-04-12 03:00:00', freq='h')
time_index_not_overlapped = pd.date_range('2000-04-05 20:00:00', '2000-06-01 18:00:00', freq='h')
time_index_one_overlap = pd.date_range('2000-01-03 09:00:00', '2000-04-12 03:00:00', freq='h')
time_index_length_one = pd.date_range('2000-02-02 09:00:00', '2000-02-02 09:00:00', freq='h')



# def empty_time_series():
#     time_index = pd.date_range('03-01-2000 10:00:00', '20-03-2000 15:00:00', freq='h')
#     ts = TimeSeries(times=time_index, values=np.full(len(time_index), np.nan))
#     return ts

def empty_values_ts(time_index):
    values = np.full(len(time_index), np.nan)
    return TimeSeries(times=time_index, values=values)

def random_values_ts(time_index):
    values = np.full(len(time_index), np.random.randint(0, 100))
    return TimeSeries(times=time_index, values=values)

# def sporadic_mv_ts(time_index, gaps=20):
#     ts = random_values_ts(time_index)
#     return create_artificial_sporadic_missing_values(ts, 20)


def create_test_series(values, offset=0, base_time='2026-01-01 12:00:00'):
    """
    Creates a Series with an hourly index.
    offset: shifts the starting hour relative to base_time (can be positive or negative)
    """
    start_time = pd.Timestamp(base_time) + pd.Timedelta(hours=offset)
    index = pd.date_range(start=start_time, periods=len(values), freq='h')
    return TimeSeries(times=index, values=values)

