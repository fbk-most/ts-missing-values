import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
import pandas as pd

from darts.dataprocessing.transformers.window_transformer import WindowTransformer

from .utility import overlap_series, overlap_series_strict, align_to_main_series
from .comparison import METRICS, TRANSFORMS


def extract_gap_density(series:TimeSeries, window_size:int=168, plot:bool=False) -> TimeSeries:
    """
        Compute the missing value density of a series using a Rolling Window.

        Parameters
        ----------
        series
            time series
        window_size
            size of the rolling window
        plot
            missing value density graph
        threshold
            number between 0 and 1, if not 0 returns only the density greater than the threshold percentage

        Returns
        -------
        TimeSeries
            gap density time series
    """
    
    if window_size<1 or not isinstance(window_size, int):
        raise ValueError('window_size must be an integer greater than 0')

    transform = {
        'function': lambda x: np.sum(np.isnan(x)),
        'mode': 'rolling',
        'window': window_size,
        'min_periods': 0,
        'center': True,
    }
    wt = WindowTransformer(transform, forecasting_safe=False)
    gap_density_series = wt.transform(series)

    if plot:
        plt.figure(figsize=(15,4))
        series.plot()
    
        plt.figure(figsize=(15,4))
        gap_density_series.plot()   
    
    return gap_density_series


def _invert_nan(x:np.ndarray) -> np.ndarray:
    """
        given a numpy array replaces:
        - np.nan with 0
        - any number with np.nan
    """
    result = x.copy()
    result[np.isnan(result)] = 0
    result[result != 0] = np.nan
    return result


def _keep_edge_values(series:TimeSeries, window:int, percentage_to_keep:float) -> TimeSeries:
    """
        gap_filter is a np.array and for each sequence of np.nan we replace the first and last bit of np.nan values with zeros
    """
    
    flag = 0
    replaced_count = 0
    gap_filter = series.values().flatten()
    percentage_to_keep = percentage_to_keep
    values_to_keep = int(window*percentage_to_keep)
    
    for i, value in enumerate(gap_filter):
        if np.isnan(value) and flag==0:
            flag=1
            
        if flag==1:
            if replaced_count <= values_to_keep:
                gap_filter[i] = 0
                replaced_count += 1
            if not np.isnan(value):
                flag = 0
                for j in range(values_to_keep):
                    gap_filter[i-j] = 0
                    replaced_count = 0

    return TimeSeries.from_times_and_values(series.time_index, gap_filter)


def gap_transform(series:TimeSeries, window_size:int=168, threshold_percentage:float=0.5, plot:bool=False, return_density_series:bool=False) -> TimeSeries | tuple[TimeSeries, TimeSeries]:
    """
        Unify areas of dense missing values (= areas with sparse values) in a TimeSeries into larger, continuous gaps based on a rolling window and density threshold.

        Parameters
        ----------
        series
            time series
        window_size
            size of the rolling window
        threshold_percentage
            a gap is unified if more than this % of values are missing
        plot
            the transformed series and the gap density graph
        return_density_series
            other than the transformed series, returns the high gap density series so only where the density series is greater than the threshold

        Returns
        -------
        TimeSeries
            transformed series with unified gaps
        TimeSeries
            high gap density series only if return_density_series=True
    """
    # threshold for detecting high density gapped area that needs to become one single large gap
    threshold = window_size * threshold_percentage
    
    gap_density_series = extract_gap_density(series, window_size, plot=plot)
    gap_density_df = gap_density_series.to_dataframe()
    high_gap_density_df = gap_density_df[gap_density_df.iloc[:,-1:] > threshold]
    high_gap_density_series = TimeSeries.from_dataframe(high_gap_density_df)
    
    if plot:
            plt.figure(figsize=(15,4))
            high_gap_density_series.plot()
            
    gap_filter_series = high_gap_density_series.map(lambda x: _invert_nan(x))
    gap_filter_series = _keep_edge_values(gap_filter_series, window_size, percentage_to_keep=0.25)
    
    gap_unified_series = series + gap_filter_series # original with unified gaps

    if return_density_series:
        return gap_unified_series, high_gap_density_series
    else:
        return gap_unified_series


def get_mean_seasonality(series:TimeSeries, freq:int) -> TimeSeries:
    """
        Return the mean seasonality of a series of a given frequency.

        Parameters
        ----------
        series
            time series
        freq
            frequency of the seasonality

        Returns
        -------
        TimeSeries
            mean seasonality of length freq

        Examples
        --------
        For an hourly series get_mean_seasonality(series, 168) computes the mean weekly seasonality because 168=24*7 is the hours in a week.
    """
    vals = series.values()
    means = []
    
    for i in range(freq):
        sliced_vals = vals[i::freq]
        if sliced_vals.size > 0 and not np.all(np.isnan(sliced_vals)):
            mean = np.nanmean(sliced_vals)
        else:
            mean = np.nan
        means.append(mean)

    day_of_week = series[0].time_index.day_of_week[0]
    hour = series[0].time_index.hour[0]
    shift = day_of_week*24+hour
    means = np.roll(means, shift)
    
    seasonality_mean = TimeSeries.from_values(values=np.array(means))
    
    return seasonality_mean


def replace_nan_with_mean_seasonality(series:TimeSeries, season_length:int) -> TimeSeries:
    """
        Replace the series NaN values with the mean_seasonality, season length rappresents the length of the seasonality to compute in order to replace the NaN values.

        Parameters
        ----------
        series
            time series
        season_length
            mean seasonality to compute in order to replace the nan values

        Returns
        -------
        TimeSeries
            time series with the NaN values replaced
    """
    
    mean_seasonality = get_mean_seasonality(series, season_length)
    
    day_of_week = series[0].time_index.day_of_week[0]
    hour = series[0].time_index.hour[0]
    shift = day_of_week*24+hour
    
    shifted_values = np.roll(mean_seasonality.values().flatten(), -shift)
    number_of_seasonalities = (len(series) // len(mean_seasonality))+1

    mean_seasonality_repeated_values = np.tile(shifted_values, number_of_seasonalities)[:len(series)] # repeated mean series
    series_values = series.values().flatten()

    mask = np.isnan(series_values)
    filled_values = np.where(mask==True, mean_seasonality_repeated_values, series_values) 

    filled_series = TimeSeries.from_times_and_values(series.time_index, filled_values)

    return filled_series


# ====================================================================
# SMALL GAP FILLING
# ====================================================================

def replace_nan_with_neighbors(arr:np.ndarray, num_values:int=5, distance:int=168) -> np.ndarray:
    """
        Replace nan values with the weighted mean of neighbouring values at a certain distance.

        Parameters
        ----------
        arr
            array to fill
        num_values
            number of neighbours to consider
        distance
            incremental distance of each neighbour from the nan value
        
        Returns
        -------
        np.ndarray
            array with the replaced values
        
        Example
        -------
        Given a nan value at index i, if distance is 24 and num_values is 3:
        the value is replced by the mean of values at index [i-72, i-48, i-24, i+24, i+48, i+72]. 
    """
    result = arr.copy()
    nan_indices = np.where(np.isnan(arr))[0]

    right_offsets = [distance * (i + 1) for i in range(num_values)]
    left_offsets = [-x for x in right_offsets][::-1]
    offsets = left_offsets + right_offsets
    
    for i in nan_indices:
        positions = i + np.array(offsets)
        valid_positions = positions[(positions >= 0) & (positions < len(arr))]
        
        if len(valid_positions) > 0:
            values = arr[valid_positions]
            non_nan_values = values[~np.isnan(values)]
            non_nan_positions = valid_positions[~np.isnan(values)]
            
            if len(non_nan_values) > 0:
                weights = 1/np.abs(non_nan_positions-i)
                weights /= weights.sum()

                result[i] = (non_nan_values*weights).sum() # in this way the values are weighted proportional to 1/d, with d the distance to the point like 1 week, 2 weeks... for distance=168
                #result[i] = np.mean(non_nan_values) # in this way the simple mean is calculated
    return result

    


def small_gap_filling(series:TimeSeries, high_gap_density_series:TimeSeries=None, num_values:int=3, distance:int=168, mean_seasonality:int=168) -> TimeSeries:
    """
        Fill all the gaps (missing values) except those in the high_gap_density_series range.

        For each missing value, a weighted average inversely proportionate to the distance is computed, taking into account the same timestamp (hour and day) but in neighboring weeks.
        Gaps that aren't successfully filled with this method (because all the neighouring vlues are mising too) are filled with the series mean seasonality.
        
        Parameters
        ----------
        series
            series to fill
        high_gap_density_series
            series rappresenting the gaps not to fill
        num_values
            number of values to use in the replace_nan_with_neighbors function
        distance
            distance to use in the replace_nan_with_neighbors function
        mean_seasonality
            the mean seasonality to use in case the main method can't fill the missing value
        
        Returns
        -------
        TimeSeries
            the time series with the 'small gaps' filled
    """

    # filling the small gaps
    df = series.to_dataframe()
    values = replace_nan_with_neighbors(df.iloc[:, 0].values, num_values, distance)
    small_gap_transformed_series = TimeSeries.from_times_and_values(series.time_index, values)
    
    # if there are still NaN values they are replaced with the mean seasonality taking into consideration 24*7=168 so the weekly seasonality
    small_gap_transformed_series = replace_nan_with_mean_seasonality(small_gap_transformed_series, mean_seasonality)
    small_gap_transformed_values = small_gap_transformed_series.values().flatten()

    if high_gap_density_series is None:
        small_gap_filled_series = TimeSeries.from_times_and_values(series.time_index, small_gap_transformed_values)
        return small_gap_filled_series
    
    # reintroducing the large gaps
    high_gap_density_values = high_gap_density_series.values().flatten()
    high_gap_density_mask = ~np.isnan(high_gap_density_values)
    filtered_values = np.where(high_gap_density_mask==False, small_gap_transformed_values, np.nan)
    small_gap_filled_series = TimeSeries.from_times_and_values(small_gap_transformed_series.time_index, filtered_values)
    
    return small_gap_filled_series



# ====================================================================
# LARGE GAP FILLING
# ====================================================================

def _distance_to_weights(distances:list[float], verbose:bool=False) -> list[float]:
    """
        Return weights inversly proportional to weights.
    """
    
    distances_inverted = [1/x for x in distances]
    isum = sum(distances_inverted)
    weights = [x/isum for x in distances_inverted]

    if verbose:
        for d, w in zip(distances, weights):
            print(d, '-',w)
    return weights

def _redistribute_weights(series: list[TimeSeries], weights: list[float]) -> np.ndarray:
    """
        Takes N weights, one for each of the N series in `series`,
        and redistributes them across timesteps so that at each
        timestamp the weights are normalized over only the series
        that have a non-NaN value at that timestamp.
    """
    
    vals = []

    for ts in series:
        vals.append(ts.values().flatten())
    
    presence_mask = np.where(~np.isnan(vals), 1, np.nan)
    
    i=0
    for v, w in zip(presence_mask, weights):
        presence_mask[i] = v*w
        i += 1
    
    stack = np.stack(presence_mask, axis=0)
    stack_sum = np.nansum(stack, axis=0)
    stack_weights = presence_mask/stack_sum
    return np.unstack(stack_weights)


def _get_filling_series(series:list[TimeSeries], distances:list[float], verbose:bool=False, return_weights:bool=False) -> TimeSeries | tuple[TimeSeries, list[float]]:
    """
        Returns the weighted sum of the series based on the distances.
        takes in consideration NaN values

        series: a list of series of the same length
        distances: a list of distances, length must be equal to the number of series
    """
    weights = _distance_to_weights(distances, verbose)
    r_weights = _redistribute_weights(series, weights)
    
    vals=[]
    for ts in series:
        vals.append(ts.values().flatten())
    stack = np.stack(vals)
    stack_weighted = stack*r_weights
    
    filling_values = np.nansum(stack_weighted, axis=0)
    all_nan = np.all(np.isnan(stack_weighted), axis=0)
    filling_values[all_nan] = np.nan

    filling_series = TimeSeries.from_times_and_values(series[0].time_index, filling_values, fill_missing_dates=True)
    
    if return_weights:
        return filling_series, weights
    return filling_series
    
def top_k_similar_series(main_series:TimeSeries, candidate_series_list:list[TimeSeries], transform:str='none', metric:str='mae', k:int=3, quality_factor:str='percentage_gap_covered') -> list[tuple[TimeSeries, float]]:
    """
        Return the k most similar series to the main_series; candidate_series_list are compared to the main series using a metric and a transform.

        Parameters
        ----------
        main_series
            the reference TimeSeries to which all candidates are compared
        candidate_series_list
            a list of TimeSeries to be evaluated for similarity
        transform
            the name of the transform to apply to both 
            the main and candidate series before comparison
            values: mae, mape, mape_symmetric, rmse, max_distance, correlation, cosine_similarity
        metric
            the name of the metric used to quantify the difference
            values: none, log, depth, percentile, diff
        k
            the number of top similar series to return
        quality_factor
            choose between prioritizing one of these indicators
            the final distance computed is divided by this factor in order to increse the distance depending on different goals
            options: percentage_overlapped_vaues, percentage_gap_covered, percentage_symmetric_overlap, num_values_used, num_values_not_used

        Returns
        -------
        list[tuple[TimeSeries, float]]
            a list of tuples, where each tuple contains the TimeSeries and its 
            calculated similarity score, sorted from most to least similar
    """
    
    transform_func = TRANSFORMS[transform]
    metric_func = METRICS[metric]
    
    main_transformed = transform_func(main_series)

    results = []
    for cand in candidate_series_list:
        
        cand_transformed = transform_func(cand)

        _, _, overlapping_stats = overlap_series_strict(main_transformed, cand_transformed, return_overlapping_stats=True)
        percentage_gap_covered = overlapping_stats[quality_factor]
        
        metric_value = metric_func(main_transformed, cand_transformed)
        if percentage_gap_covered != 0:
            results.append((cand, metric_value/percentage_gap_covered))
    
    results.sort(key=lambda x: x[1])
    
    return results[1:k+1]


def large_gap_filling(series:TimeSeries, high_gap_density_series:TimeSeries, candidates:list[TimeSeries], transform:str='none', metric:str='mae', k:int=3, plot_weighted_portions:bool=False, return_filling_series:bool=False) -> TimeSeries | tuple[TimeSeries, TimeSeries]:
    """
        Fill large gaps of a series using the KNN algorithm.

        Must provide the series to fill, a high gap density series showing the large gaps to fill, the candidate series for the filling.

        Parameters
        ----------
        series
            the series to fill
        high_gap_density_series
            the series that identifies the large gaps
        candidates
            a list of TimeSeries to be evaluated for similarity
            do NOT include the series to fill in this list
        transform
            the name of the transform to apply to both 
            the main and candidate series before comparison
            values: mae, mape, mape_symmetric, rmse, max_distance, correlation, cosine_similarity
        metric
            the name of the metric used to quantify the difference
            values: none, log, depth, percentile, diff
        k
            the number of top similar series to return
        plot_weighted_portions
            plot each of the k time series used for the filling with relative weights
        return_filling_series
            other than the filled time series, reutrn also the filling series

        Returns
        -------
        TimeSeries
            the time series with filled large gaps
        tuple[TimeSeries, TimeSeries]
            the time series with filled large gaps and the filling series
    """
    
    high_gap_density_values = high_gap_density_series.values().flatten()
    high_gap_density_mask = ~np.isnan(high_gap_density_values)
    
    candidates_and_distances = top_k_similar_series(series, candidates, transform, metric, k)
    candidates = [candidate for candidate, distance in candidates_and_distances]
    distances = [distance for candidate, distance in candidates_and_distances]
    
    filling_series = TimeSeries.from_times_and_values(high_gap_density_series.time_index, np.zeros(len(high_gap_density_series)))
    candidate_series_overlapped = align_to_main_series(high_gap_density_series, candidates)
    
    filling_series_whole, weights = _get_filling_series(candidate_series_overlapped, distances, verbose=False, return_weights=True)
    filling_values = np.where(high_gap_density_mask==True, filling_series_whole.values().flatten(), 0)
    filling_series = TimeSeries.from_times_and_values(high_gap_density_series.time_index, filling_values)

    if plot_weighted_portions:
        i=0
        for ts, weight in zip(candidate_series_overlapped, weights):
            plt.figure()
            ts.plot(alpha=0.5, label='series')
            
            ts_filling_values = np.where(high_gap_density_mask==True, ts.values().flatten(), np.nan)
            ts_filling = TimeSeries.from_times_and_values(high_gap_density_series.time_index, ts_filling_values)
            
            (ts_filling).plot(color='black', label='portion')
            (ts_filling*weight).plot(color='blue', label='weighted portion')
    
        plt.figure()
        filling_series.plot(label='filling series')
        print('number of missing values in the filling series: ',np.isnan(filling_series.values()).sum())
    
    series_values_with_zeros = np.where(high_gap_density_mask==True, 0, series.values().flatten()) 
    series_to_fill_with_zeros = TimeSeries.from_times_and_values(series.time_index, series_values_with_zeros)
    
    filled_series = (series_to_fill_with_zeros + filling_series)

    if return_filling_series:
        return filled_series, filling_series
    return filled_series


# ====================================================================
# ALL GAPS FILLING
# ====================================================================

def universal_filling(series:TimeSeries, candidates:list[TimeSeries], num_values:int=3, distance:int=168, mean_seasonality:int=168, transform:str='none', metric:str='mae', k:int=3, return_large_gap_filled_series:bool=False) -> TimeSeries:
    """
        Fills all the gaps of a series.

        Parameters
        ----------
        series
            the series with missing values to fill
        candidates
            list of TimeSeries from which the function will choose the best candidates
            do NOT include the series to fill in this list
        num_values
            number of values to use in the replace_nan_with_neighbors function
        distance
            distance to use in the replace_nan_with_neighbors function
        mean_seasonality
            the mean seasonality to use in case the main method can't fill the missing value
        transform
            the name of the transform to apply to both 
            the main and candidate series before comparison
            values: mae, mape, mape_symmetric, rmse, max_distance, correlation, cosine_similarity
        metric
            the name of the metric used to quantify the difference
            values: none, log, depth, percentile, diff
        k
            the number of top similar series to return

        Returns
        -------
        TimeSeries
            the fully filled time series
    """
    _, high_gap_density_series  = gap_transform(series, window_size=168, threshold_percentage=0.5, plot=False, return_density_series=True)
    small_gap_filled_series = small_gap_filling(series, high_gap_density_series, num_values=num_values, mean_seasonality=mean_seasonality, distance=distance)
    large_gap_filled_series, filling_series = large_gap_filling(small_gap_filled_series, high_gap_density_series, candidates, return_filling_series=True, transform=transform, metric=metric, k=k)

    # returning the series before the final step of small gap filling, this could result in some cases in a non fully-filled series
    if return_large_gap_filled_series:
        return large_gap_filled_series
    
    # running small gap filling again in case the large gap filling series has still some missing values
    filled_series = small_gap_filling(large_gap_filled_series, high_gap_density_series=None, num_values=num_values, mean_seasonality=mean_seasonality, distance=distance)
    
    return filled_series



# ====================================================================
# GAP FILLING VALIDATION
# ====================================================================


def random_hour_timestamp(start, end, unit='hours'):    
    n_hours = int((end - start) / pd.Timedelta(1, unit))
    rand_hours = np.random.randint(0, n_hours + 1)

    return start + pd.Timedelta(rand_hours, unit)


def create_artificial_large_gap(series, gap_length=24*30, unit='hours', return_removed_slice=False, plot=False):
    '''
    Deletes randomly values from a time series creating an artificial continous gap.
    The gap ensures all values are NaN in that range.
    
    :param series: time series
    :param gap_length: number of continous values to replace with NaN
    :param unit: pd time unit (like 'hours' 'minutes' 'days'...) which rappresents the frequency of the time series
    :param return_gap: returns the 
    :param plot: Description
    '''
    if gap_length >= len(series):
        gap_length = len(series)-5

    start_gap_time = random_hour_timestamp(
        start = series.start_time(),
        end = series.end_time() - pd.Timedelta(gap_length+1, unit)
    )
    end_gap_time = start_gap_time + pd.Timedelta(gap_length, unit)

    _, slice_to_remove = series.split_before(start_gap_time)
    slice_to_remove, _ = slice_to_remove.split_before(end_gap_time)

    artificial_gap_values = np.full(len(slice_to_remove), np.nan)
    artificial_gap = TimeSeries.from_times_and_values(slice_to_remove.time_index, artificial_gap_values)

    series_before_slice, _ = series.split_before(start_gap_time)
    _, series_after_slice = series.split_before(end_gap_time)

    # check edge cases and create gapped time series
    if start_gap_time == series.start_time():
        series_with_gap = series_after_slice
    elif end_gap_time == series.end_time():
        series_with_gap = series_before_slice
    else:
        series_with_gap = series_before_slice.concatenate(artificial_gap)
        series_with_gap = series_with_gap.concatenate(series_after_slice)

    if plot==True:
        plt.figure(figsize=(15,4))
        series.plot(label='base series')
        slice_to_remove.plot(label='slice to remove')
        plt.figure(figsize=(15,4))
        series_with_gap.plot(label='series with artifical gap')
    
    if return_removed_slice:
        return series_with_gap, slice_to_remove
 
    return series_with_gap
    