import pytest
from conftest import *
from ts_missing_values.preprocessing import *


# ---------- series to test factory ----------

def series_with_outliers():
    v = [1,2,3]*100 + [1200] + [1,2,np.nan]*100 + [-1200]
    return create_test_series(v)

def full_series():
    v = [*range(-10, 10)]*10
    return create_test_series(v)

def nan_series():
    v = [np.nan]*100
    return create_test_series(v)

def short_series():
    v = [1]
    return create_test_series(v)

def surrounded_by_nans_series():
    v = [np.nan]*5 + [1,2,3] + [np.nan]*5
    return create_test_series(v)

def head_series():
    v = [1,2,3] + [np.nan]*5
    return create_test_series(v)

def tail_series():
    v = [np.nan]*5 + [1,2,3]
    return create_test_series(v)

# --------------------------------------


@pytest.mark.parametrize(['ts_in', 'expected'], [
    [series_with_outliers(), np.array([1,2,3]*100 + [np.nan] + [1,2,np.nan]*100 + [-1200])],
    [nan_series(),           np.array([np.nan]*100)],
    [full_series(),          np.array([*range(-10, 10)]*10)],
    [short_series(),         np.array([1])],
])
def test_eliminate_outliers(ts_in, expected):
    ts_out = eliminate_outliers(ts_in)

    assert ts_in.shape == ts_out.shape
    assert np.array_equal(ts_out.univariate_values(), expected, equal_nan=True)


def test_eliminate_outliers_values_symmetric():
    ts_in = series_with_outliers()

    ts_out = eliminate_outliers(ts_in, symmetric=True)
    out_values = ts_out.univariate_values()
    assert np.array_equal(out_values, np.array([1,2,3]*100 + [np.nan] + [1,2,np.nan]*100 + [np.nan]), equal_nan=True)


def test_eliminate_outliers_verbose(capsys):
    ts_in = series_with_outliers()
    ts_in = ts_in.with_metadata({'name': 'TestTestSeries'})

    ts_out = eliminate_outliers(ts_in, verbose=True)
    out_values = ts_out.univariate_values()
    assert np.array_equal(out_values, np.array([1,2,3]*100 + [np.nan] + [1,2,np.nan]*100 + [-1200]), equal_nan=True)

    captured = capsys.readouterr()
    assert "TestTestSeries" in captured.out
    assert "number of outliers deleted" in captured.out


def test_eliminate_outliers_verbose_no_metadata(capsys):
    ts_in = series_with_outliers()

    ts_out = eliminate_outliers(ts_in, verbose=True)
    out_values = ts_out.univariate_values()
    assert np.array_equal(out_values, np.array([1,2,3]*100 + [np.nan] + [1,2,np.nan]*100 + [-1200]), equal_nan=True)

    captured = capsys.readouterr()
    assert not "TestSeries" in captured.out
    assert "number of outliers deleted" in captured.out


@pytest.mark.parametrize(['ts_in', 'expected'], [
    [series_with_outliers(), series_with_outliers()],
    [nan_series(),           nan_series()],
    [full_series(),          full_series()],
    [short_series(),         short_series()],
])
def test_preprocess_series(ts_in, expected):
    ts_out = preprocess_series(
        ts_in,
        delete_outliers = False,
        symmetric = False,
        log_scale = False,
        interpolate_missing_values = False,
        verbose = False
        )
    assert ts_out == expected


@pytest.mark.parametrize(['ts_in', 'expected'], [
    [series_with_outliers(), np.array([1,2,3]*100 + [np.nan] + [1,2,np.nan]*100 + [-1200])],
    [nan_series(),           np.array([np.nan]*100)],
    [full_series(),          np.array([*range(-10, 10)]*10)],
    [short_series(),         np.array([1])],
])
def test_preprocess_series_outliers(ts_in, expected):
    ts_out = preprocess_series(
        ts_in,
        delete_outliers = True,
        symmetric = False,
        log_scale = False,
        interpolate_missing_values = False,
        verbose = False
        )
    out_values = ts_out.univariate_values()
    assert np.array_equal(out_values, expected, equal_nan=True)



def test_preprocess_series_log_scale():
    ts_in = series_with_outliers()
    ts_out = preprocess_series(
        ts_in,
        delete_outliers = False,
        symmetric = False,
        log_scale = True,
        interpolate_missing_values = False,
        verbose = False
        )
    out_values = ts_out.univariate_values()
    
    vals = ts_in.univariate_values()
    vals[vals<0] = np.e
    out_should_be = np.log(vals)

    assert np.array_equal(out_values, out_should_be , equal_nan=True)


def test_preprocess_series_log_scale_nan():
    ts_in = nan_series()
    ts_out = preprocess_series(
        ts_in,
        delete_outliers = False,
        symmetric = False,
        log_scale = True,
        interpolate_missing_values = False,
        verbose = False
        )
    out_values = ts_out.univariate_values()
    out_should_be = [np.nan]*100

    assert np.array_equal(out_values, out_should_be , equal_nan=True)


@pytest.mark.parametrize(['ts_in'], [
    [series_with_outliers()],
    [full_series()],
    [short_series()],
    [surrounded_by_nans_series()],
    [head_series()],
    [tail_series()]
])
def test_preprocess_series_interpolate(ts_in):
    ts_out = preprocess_series(
        ts_in,
        delete_outliers = False,
        symmetric = False,
        log_scale = False,
        interpolate_missing_values = True,
        verbose = False
        )
    out_values = ts_out.univariate_values()

    assert np.isnan(out_values).sum() == 0


def test_preprocess_series_interpolate_nans():
    ts_in = nan_series()
    ts_out = preprocess_series(
        ts_in,
        delete_outliers = False,
        symmetric = False,
        log_scale = False,
        interpolate_missing_values = True,
        verbose = False
        )
    out_values = ts_out.univariate_values()

    assert np.isnan(out_values).sum() == len(ts_in)