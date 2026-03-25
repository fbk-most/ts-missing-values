import numpy as np
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values


def eliminate_outliers(series:TimeSeries, N:int=3, symmetric:bool=False, verbose:bool=False) -> TimeSeries:
    """
        Eliminate outliers greater than 'mean + N*stdev'.

        This method is best used on stationary time series or series without a significant trend.

        Parameters
        ----------
        series
            the time series
        N
            number of standard deviations to consider for the threshold
        symmetric
            if True eliminates also values lower than 'mean - N*stdev'
        
        Returns
        -------
        TimeSeries
            series having ouliers replaced with np.nan
    """

    values = series.univariate_values()
    values_non_nan = values[~np.isnan(values)]

    stdev = np.std(values_non_nan)
    mean = np.mean(values_non_nan)
    
    # replace low/high values with NaN
    values_without_outliers = np.array([v if v < mean + N*stdev else np.nan for v in values])
    if symmetric:
        values_without_outliers = np.array([v if v > mean - N*stdev else np.nan for v in values_without_outliers])
    
    #sereis_without_outliers = TimeSeries.from_times_and_values(series.time_index, values_without_outliers)
    sereis_without_outliers = series.with_values(values_without_outliers)
    if verbose:
        old_val = series.univariate_values()
        new_val = sereis_without_outliers.univariate_values()
        outliers_deleted = np.isnan(new_val).sum() - np.isnan(old_val).sum()
        percentage_change = 100 - 100 * ((~np.isnan(new_val)).sum() / (~np.isnan(old_val)).sum())
        
        if series.has_metadata and ('name' in series.metadata):
            print(f"{series.metadata['name']:20} number of outliers deleted: {outliers_deleted:10} = {percentage_change:5.2f}%")
        else:
            print(f"number of outliers deleted: {outliers_deleted:10} = {percentage_change:5.2f}%")
    return sereis_without_outliers


def preprocess_series(series:TimeSeries, delete_outliers:bool=False, N:int=3, symmetric:bool=False, log_scale:bool=False, interpolate_missing_values:bool=False, verbose:bool=False) -> TimeSeries:     
    """
        Simple preprocessing of a series with 3 main boolean options: outliers, logarithmic scale, fill missing values.

        Parameters
        ----------
        delete_outliers
            detect and eliminate outliers using a threshold
            for N and symmetric: see eliminate_outliers()
        log_scale
            change scale of the series into logarithmic scale (base e)
        interpolate_missing_values
            this uses interpolation in order to fill gaps, useful for simple cases or small gaps, DON'T use it for complex series 
            this converts all non positive values to 1 then transforms the series into log scalse;
            this is is done in order to make the log transform possible and it works well for traffic data (all values >=0).
        verbose
            print the amount of outliers detected

        Returns
        -------
        TimeSeries
            preprocessed series with the cosen options
    """

    result_series = series
    
    if delete_outliers:
        result_series = eliminate_outliers(result_series, N, symmetric=symmetric, verbose=verbose)
    
    if interpolate_missing_values:
        result_series = fill_missing_values(result_series)

    if log_scale:
        series_log = result_series.map(np.log)
        vals = series_log.values()
        vals[vals<=0]=1
        
        series_wo_zeros = TimeSeries.from_times_and_values(result_series.time_index, vals)
        result_series = series_wo_zeros

    return result_series