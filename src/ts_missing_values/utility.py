import numpy as np
from darts import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd
from darts.utils.utils import generate_index
from typing import Union
from pandas.tseries.frequencies import to_offset

def generate_periodic_array(periods:int=5, seasonality:int=10, noise:bool=True, plot:bool=False, noise_level:float=0.6) -> np.ndarray:
    """
        Generate a simple periodic numpy array with noise for test purposes.

        Each period is monotonic and grows from 0 to freq.

        Parameters
        ----------
        periods
            number of periods
        seasonality
            length of each period
        noise
            add gaussian noise to the array
        noise_level
            level of gaussian noise, the higher the number the greater the noise
        plot
            graph of the series

        Returns
        -------
        np.ndarray
            periodic array
    """
    if periods<1:
        raise ValueError('number of periods n muse be an integer greater than 1')

    arr = np.array([*range(seasonality)]*periods, dtype=float)

    if noise:
        random_noise = np.random.normal(0, noise_level, size=len(arr))
        arr += random_noise

    if plot:
        plt.plot(arr)

    return arr


def generate_periodic_series(periods:int=5, seasonality:int=5, freq:str='h', start:pd.Timestamp=None, plot:bool=False, noise:bool=True, noise_level:float=0.6) -> TimeSeries:
    """
        Generate a simple periodic Time series with noise for test purposes.
        
        Parameters
        ----------
        periods
            number of periods
        seasonality
            length of each period
        freq
            from start, the frquency where the values are mapped in pandas format, e.g. 'h' is hourly or 'D' for daily
            https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        noise
            add gaussian noise to the array
        noise_level
            level of gaussian noise, the higher the number the greater the noise
        plot
            graph of the series

        Returns
        -------
        TimeSeries
            periodic series
    """
    if periods<1:
        raise ValueError('number of periods n muse be an integer greater than 1')

    arr = generate_periodic_array(periods=periods, seasonality=seasonality, noise=noise, noise_level=noise_level)
    ts = None
    if start is None:
        ts = TimeSeries.from_values(arr)
    else:
        time_index = pd.date_range(start=start, freq=freq, periods=len(arr))
        ts = TimeSeries(time_index, arr)
    
    if plot:
        ts.plot()
    
    return ts


def extend_series(series:TimeSeries, new_start:Union[pd.Timestamp, str]=None, new_end:Union[pd.Timestamp, str]=None) -> TimeSeries:
    """
        Return series extended to the new start and end with np.nan .

        Parameters
        ----------
        series
            the series to extend
        new_start
            new starting timestamp, can also be a string
        new_end
            new ending timestamp, can also be a string
        
        Return
        ------
        TimeSeries
            extended time series
    """

    series_result = series
    freq = series.freq_str
    component_count = series.width
    sample_count = series.n_samples
    dtype = series.dtype

    if new_start is not None:
        new_start_timestamp = pd.to_datetime(new_start)
        start = series.start_time()
        before_index = generate_index(start=new_start_timestamp, end=start - series.freq, freq=freq)
        if len(before_index) > 0:
            before_values = np.full((len(before_index), component_count, sample_count), np.nan, dtype=dtype)
            before_series = TimeSeries.from_times_and_values(times=before_index, values=before_values)
            series_result = before_series.append(series_result)

    if new_end is not None:
        new_end_timestamp = pd.to_datetime(new_end)
        end = series.end_time()
        after_index = generate_index(start=end + series.freq, end=new_end_timestamp, freq=freq)
        if len(after_index) > 0:
            after_values = np.full((len(after_index), component_count, sample_count), np.nan, dtype=dtype)
            after_series = TimeSeries.from_times_and_values(times=after_index, values=after_values)
            series_result = series_result.append(after_series)
    series_result = series_result.with_metadata(series.metadata)
    return series_result


def _are_overlappable(series_A:TimeSeries, series_B:TimeSeries) -> bool:
    start1 = series_A.start_time()
    start2 = series_B.start_time()
    
    end1 = series_A.end_time()
    end2 = series_B.end_time()

    if end2<start1 or start2>end1:
        return False
    return True

def overlap_series(series_A:TimeSeries, series_B:TimeSeries) -> tuple[TimeSeries, TimeSeries]:
    """
        Return two subseries taking only the overlapping range of time_index.

        Parameters
        ----------
        series_A
            the first series
        series_B
            the second series
        
        Returns
        -------
        TimeSeries, TimeSeries
            both the series overlapped
    """
    if not _are_overlappable(series_A, series_B):
            raise ValueError('The two time series dont\'t have the same time range and are not overlappable')
    
    res_A = series_A.slice_intersect(series_B)
    res_B = series_B.slice_intersect(series_A)
    
    return res_A, res_B


def overlap_series_strict(series_A:TimeSeries, series_B:TimeSeries, return_overlapping_stats:bool=False) -> tuple[TimeSeries, TimeSeries]:
    """
        Return the perfectly overlapped series: the overlap is valid only if both series have a non-NaN value for a timestamp.
        
        The values are compacted so the time index has no intresting value and should not be taken in consideration.
        The indexing is used only for a comparison.

        Parameters
        ----------
        series_A
            the first series
        series_B
            the second series
        
        Returns
        -------
        If return_overlapping_stats is False (default):
        TimeSeries, TimeSeries
            both the series strictly overlapped

        If return_overlapping_stats is True:
        TimeSeries, TimeSeries, dict
            can return also the percentage of values of the first series that were used in the overlap.
    """

    if not _are_overlappable(series_A, series_B):
            raise ValueError('The two time series dont\'t have the same time range and are not overlappable')
    
    start = min(series_A.start_time(), series_B.start_time())
    end = max(series_A.end_time(), series_B.end_time())
    #num_values_total_range = (end-start).total_seconds()/60/60
    if series_A.freq == 1:
        num_values_total_range = end-start
    else:
        num_values_total_range = len(pd.date_range(start,end,freq=series_A.freq))
    
    
    series_A_overlapped, series_B_overlapped = overlap_series(series_A, series_B)
    values_A = series_A_overlapped.values().flatten()
    values_B = series_B_overlapped.values().flatten()
    
    mask_overlap = ~np.isnan(values_A) & ~np.isnan(values_B)
    values_A_overlapped = values_A[mask_overlap==True]
    values_B_overlapped = values_B[mask_overlap==True]

    mask_gap = np.isnan(values_A) & ~np.isnan(values_B)
    values_A_gap = values_A[mask_gap==True] # where the first series has nan values
    values_B_gap = values_B[mask_gap==True] # and the second one has numeric values
    
    ts_A = TimeSeries.from_times_and_values(series_A_overlapped.time_index[:len(values_A_overlapped)], values_A_overlapped)
    ts_B = TimeSeries.from_times_and_values(series_A_overlapped.time_index[:len(values_A_overlapped)], values_B_overlapped)

    # calculate and return statistics of the overlap
    if return_overlapping_stats:
        original_values = series_A.values().flatten()
        num_nan_values = np.count_nonzero(np.isnan(original_values))
        num_values = len(original_values) - num_nan_values
        num_values_overlapped = len(values_A_overlapped)
        
        if num_values > 0:
            percentage_overlapped_values = num_values_overlapped / num_values
        else:
            percentage_overlapped_values = 0.0

        # if there is no gap to cover 1 is returned always
        if num_nan_values!=0:
            percentage_gap_covered = len(values_A_gap)/num_nan_values
        else:
            percentage_gap_covered = 1

        if num_values_total_range > 0:
            percentage_symmetric_overlap = num_values_overlapped / num_values_total_range
        else:
            percentage_symmetric_overlap = 0.0

        stats = {   
            "num_values_used": num_values_overlapped,
            "num_values_not_used": num_values-num_values_overlapped,
            "percentage_overlapped_vaues": percentage_overlapped_values,
            "percentage_gap_covered": percentage_gap_covered,
            "percentage_symmetric_overlap": percentage_symmetric_overlap,
        }
        return ts_A, ts_B, stats
        
    return ts_A, ts_B


def get_extended_series(all_series:list[TimeSeries]) -> list[TimeSeries]:
    """
        Extends a list of series before and after with NaN in order to allign all series on the same time index.

        Parameters
        ----------
        all_series
            the list of series to allign
        
        Returns
        -------
        list[TimeSeries]
            all series extended and alligned
    """
    time_index_types = [s.has_datetime_index for s in all_series]

    if not any(time_index_types):
        return _extend_series_without_datetime_index(all_series)
    elif all(time_index_types):
        return _extende_series_with_datetime_index(all_series)
    else:
        raise ValueError("All the series must have the same type of index")


def _extende_series_with_datetime_index(all_series):
    # helper function to manage the series with a date time index: in this case all series are aligned to the first start and last end
    start_times, end_times = [], []
    for s in all_series:
        start_times.append(s.start_time())
        end_times.append(s.end_time())
    firts_start = min(start_times)
    last_end = max(end_times)
    
    all_series_extended=[extend_series(s, firts_start, last_end) for s in all_series]
    return all_series_extended

def _extend_series_without_datetime_index(all_series):
        # helper function to manage the series without a date time index: in this case all series are extended to have the same length
        all_series_extended=[]
        max_length = max([len(s) for s in all_series])
        for s in all_series:
            if len(s)<max_length:
                vals = s.univariate_values()
                new_vals = np.append(vals, [np.nan]*(max_length-len(s)))
                new_s = TimeSeries.from_values(new_vals)
                all_series_extended.append(new_s)
            else:
                all_series_extended.append(s)
        return all_series_extended


def align_to_main_series(main_series:TimeSeries, other_series:list[TimeSeries]) -> list[TimeSeries]:
    """
        Given a main series, extend or cuts the other series so that they all allign to the main series.
        
        Parameters
        ----------
        main_series
            the main series

        other_series
            the other series to overlap
        
        Returns
        -------
        list[TimeSeries]
            all the other series overlapped
    """

    other_series_extended = get_extended_series([*other_series, main_series])[:-1]
    other_series_overlapped = [ts.slice_intersect(main_series) for ts in other_series_extended]
    
    return other_series_overlapped


def get_mean_series(all_series: list[TimeSeries]) -> TimeSeries:
    """
        Return the maan series of a list of seris.
       
        Parameters
        ----------
        all_series
            the list of series
        
        Returns
        -------
        TimeSeries
            the maan of all time series
    """

    all_series_extended = get_extended_series(all_series=all_series)
    
    stacked = np.vstack([ts.values().flatten() for ts in all_series_extended])
    mean = np.nanmean(stacked, axis=0)
    series_mean = TimeSeries.from_times_and_values(all_series_extended[0].time_index, mean)
    return series_mean


def extract_most_common_freq(index:pd.DatetimeIndex, return_timedelta:bool=False):
    """
        Infer the most common frequency in a datetime index.

        After sorting the index a difference between the index and itself shifted by 1 is done.
        The most common difference is returned.

        Parameters
        ----------
        index 
            Datetime index
        return_timedelta
            if true returns a timestmp instead of the offset string
            for example `h` and 0 days 01:00:00 for an hourly frequency

        Returns
        -------
        pandas.tseries.offsets.BaseOffset or pd.Timestamp or None
            The most common frequency as a pandas offset, or ``None`` if it
            cannot be determined (e.g. fewer than 2 timestamps).
    """

    if len(index) < 2:
        return None
    
    index=index.sort_values()
    diffs = index.diff(1).dropna()

    if diffs.empty:
        return None
    
    values, counts = np.unique(diffs.to_list(), return_counts=True)
    most_common_frequency = values[np.argmax(counts)]

    if return_timedelta:
        return most_common_frequency
    return to_offset(most_common_frequency)


def is_series_empty(ts:TimeSeries) -> bool:
    """
        Check if a univariate series contains only NaN values.

        Parameters
        ----------
        ts
            Time series to check
    
        Returns
        -------
        True if series is empty, False otherwise
    """
    return not bool(np.sum(~np.isnan(ts.univariate_values())))
