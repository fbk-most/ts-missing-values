from .utility import overlap_series, overlap_series_strict, get_extended_series
from sklearn.metrics.pairwise import cosine_similarity
from darts import TimeSeries
from darts.metrics.metrics import mae, mape, rmse
import numpy as np
from darts.dataprocessing.transformers.diff import Diff

from .utility import get_mean_series, is_series_empty
from .preprocessing import preprocess_series


# ====================================================================
# METRICS
# ====================================================================

METRICS = {
    "mae": lambda ts_A, ts_B: mae(ts_A, ts_B),
    "mape": lambda ts_A, ts_B: mape(ts_A, ts_B),
    "mape_symmetric": lambda ts_A, ts_B: mape_symmetric(ts_A, ts_B),
    "rmse": lambda ts_A, ts_B: rmse(ts_A, ts_B),
    "max_distance": lambda ts_A, ts_B: series_max_distance(ts_A, ts_B),
    "pearson_dissimilarity": lambda ts_A, ts_B: 1-series_correlation(ts_A, ts_B),
    "cosine_dissimilarity": lambda ts_A, ts_B: 1-series_cosine_similarity(ts_A, ts_B)
}

def series_correlation(series_A:TimeSeries, series_B:TimeSeries) -> float:
    """
        Correlation between two time series.
       
        Only where both the series have a valid value (non NaN) are used for the computation.

        Parameters
        ----------
        series_A
            the first time series
        series_B
            the second time series
        
        Returns
        -------
        float
            correlation between the two series
    """

    series_A, series_B = overlap_series_strict(series_A, series_B)

    return np.corrcoef(series_A.values().flatten(), series_B.values().flatten())[0,1]

def series_mean_euclidean_distance(series_A:TimeSeries, series_B:TimeSeries) -> float:
    """
        Mean RMSE of two series.
       
        Parameters
        ----------
        series_A
            the first time series
        series_B
            the second time series
        
        Returns
        -------
        float
            mean RMSE
    """
    return np.sqrt(np.mean((series_A.values() - series_B.values())**2))

def series_max_distance(series_A:TimeSeries, series_B:TimeSeries) -> float:
    """
        Maximum difference between two time series.
       
        Parameters
        ----------
        series_A
            the first time series
        series_B
            the second time series
        
        Returns
        -------
        float
            max distance
    """
    return np.nanmax(np.abs(series_A.slice_intersect_values(series_B) - series_B.slice_intersect_values(series_A)))

def series_cosine_similarity(series_A:TimeSeries, series_B:TimeSeries) -> float:
    """
        Cosine similarity between the two series.

        Only where both the series have a valid value (non NaN) are used for the computation.

        Parameters
        ----------
        series_A
            the first time series
        series_B
            the second time series
        
        Returns
        -------
        float
            cosine similarity
    """
    ts_A, ts_B = overlap_series_strict(series_A, series_B)
    similarity = cosine_similarity(ts_A.values().flatten().reshape(1,-1), ts_B.values().flatten().reshape(1,-1))
    return similarity[0][0]

def mape_symmetric(series_A:TimeSeries, series_B:TimeSeries, method:str='mean') -> float:
    """
        Return the maan series of a list of seris.
       
        Parameters
        ----------
        series_A
            the first time series
        series_B
            the second time series
        method
            'mean': compute the mean series between the two input series, then compute the mape between series_A and the mean one;
            'max': the ratio between the largest value and the smallest at any time index is computed     
            'min': the ratio between the smallest value and the largest at any time index is computed     

        Returns
        -------
        float
            cosine similarity
    """
    if method=='mean':
        series_A, series_B = get_extended_series([series_A, series_B])
        
        values_A = series_A.values().flatten()
        values_B = series_B.values().flatten()
        mean_values = (values_A + values_B) / 2
        ts_mean = TimeSeries.from_times_and_values(series_A.time_index, mean_values)
    
        # in this case it's independant calculating the mape with series_A or series_B cause we're comparing it to the mean series
        return mape(ts_mean, series_A)
    
    series_A, series_B = overlap_series_strict(series_A, series_B)
    values_A = series_A.values().flatten()
    values_B = series_B.values().flatten()
    numerator = np.abs(values_A - values_B)

    if method=='max':
        denominator = np.maximum(values_A, values_B)
    elif method=='min':
        denominator = np.minimum(values_A, values_B)
    else:
        raise ValueError('method should mean, max or min')
        
    return np.mean(numerator/denominator)

def compare_series(series_A:TimeSeries, series_B:TimeSeries, metric:str='mae', quality_stat:str=None) -> float:
    """
    Calculates the distance (dissimilarity) between two TimeSeries and 
    optionally applies a penalty factor based on the quality stat.

    The penalty factor is calculated by dividing the distance by a percentage
    (0 to 1), effectively increasing the distance when the overlap isn't perfect.

    Parameters
    ----------
    series_A
        The first time series
    series_B
        The second time series
    metric
        The base distance metric to use 'mae', 'mape', 'mape_symmetric', 'rmse'
    quality_stat
        The overlap statistic used as the penalty factor denominator.
        Must be one of: 'percentage_overlapped_vaues', 'percentage_gap_covered' or 'percentage_symmetric_overlap'

    Returns
    -------
    float
        The calculated distance
    """

    ALLOWED_METRICS = {'mae', 'mape', 'mape_symmetric', 'rmse'}
    ALLOWED_STATS = {'percentage_overlapped_vaues', 'percentage_gap_covered', 'percentage_symmetric_overlap'}

    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Invalid metric '{metric}'. Must be one of: {ALLOWED_METRICS}")

    if quality_stat is not None and quality_stat not in ALLOWED_STATS:
        raise ValueError(f"Invalid quality_stat '{quality_stat}'. Must be None or one of: {ALLOWED_STATS}")

    distance = METRICS[metric](series_A, series_B)

    if quality_stat is not None:
        _, _, stats = overlap_series_strict(series_A, series_B, return_overlapping_stats=True)
        return distance/stats[quality_stat]
    
    return distance


# ====================================================================
# TRANSFORMS
# ====================================================================

TRANSFORMS = {
    "none": lambda ts: ts,
    "log": lambda ts: preprocess_series(ts, delete_outliers=False, log_scale=True),
    "depth": lambda ts: median_depth_transform(ts),
    "percentile": lambda ts, num_bins=25: percentile_transform(ts, num_bins),
    "diff": lambda ts, diff_lags=1: Diff(lags=diff_lags).fit_transform(ts),
    #"mean_diff": lambda ts, mean_series=mean_series: (ts-mean_series.slice_intersect(ts)),
}


def percentile_transform(series:TimeSeries, num_bins:int=7) -> TimeSeries:
    """
        Approximate each value with it's percentile bin.

        The series is normalized between 0 and 1.
        This transformation is good to get the shape of the series indipendently of the absolute values.

        Parameters
        ----------
        series
            series to transform
        num_bins
            number of approximation values

        Returns
        -------
        TimeSeries
            Binned Series.
    """
    if is_series_empty(series):
        return series
    
    values = series.values().flatten()
    valid_indices = ~np.isnan(values)
    clean_values = values[valid_indices]
    
    transformed_values = np.full_like(values, np.nan, dtype=float)
    values_bins = np.unique(np.percentile(clean_values, np.linspace(0, 100, num_bins)))

    indices = np.digitize(clean_values, values_bins) - 1
    transformed_values[valid_indices] = indices / (len(values_bins) - 1)
    transformed_series = series.with_values(transformed_values)
    
    return transformed_series


def median_depth_transform(series:TimeSeries) -> TimeSeries:
    """
        Approximate each value with it's distance to the median value.

        The series is normalized between -0.5 and 0.5. The median has value 0.
        This transformation is good to get the shape of the series indipendently of the absolute values.

        Parameters
        ----------
        series
            series to transform

        Returns
        -------
        TimeSeries
            Transformed Series.
    """
    if is_series_empty(series):
        return series
    
    values = series.values().flatten()
    
    median = np.nanmedian(values)
    minval = np.nanmin(values)
    maxval = np.nanmax(values)
    
    sorted_values = np.sort(values[~np.isnan(values)])
    num_values = len(sorted_values)

    # find the first and last occurence index in the sorted array
    # for [1,1,1,2,2,3,4] and looking for 1 we have min_index=0 and max_index=2
    min_index = {v: np.where(sorted_values == v)[0][0] for i, v in enumerate(sorted_values)}
    max_index = {v: np.where(sorted_values == v)[0][-1] for i, v in enumerate(sorted_values)}
    median_index = num_values//2

    # associate every value with it's mean distance in respect to the median
    values_index = {
        float(v): float((min_index[v]+max_index[v]-2*median_index)/2 if v < median else (min_index[v]+max_index[v]-2*median_index)/2)
        for v in sorted_values
    }
    
    max_distance = max(abs([*values_index.values()][0]), [*values_index.values()][-1])
    values_index[median] = float(np.int64(0))
    for i in values_index:
        values_index[i] /= max_distance*2
    
    depth_values = [values_index[x] if not np.isnan(x) else np.nan for x in values]
    depth_series = TimeSeries.from_times_and_values(series.time_index, depth_values)
    return depth_series


def mean_transform(all_series:list[TimeSeries]) -> tuple[TimeSeries, list[TimeSeries]]:
    """
        Compute the mean of all series computed by taking the mean value at each time stamp.
 
        Parameters
        ----------
        all_series
            list of series to transform

        Returns
        -------
        TimeSeries, list[TimeSeries]
            mean series, all series extended with nan so that they have the same time index of the mean series
    """
    
    all_series_extended = get_extended_series(all_series)
    series_mean = get_mean_series(all_series_extended)

    return series_mean, all_series_extended

