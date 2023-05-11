import numpy as np
import time
import pytest
import threading

from qutip.solver.parallel import (
    parallel_map, serial_map, loky_pmap, MapExceptions
)


def _func1(x):
    return x**2


def _func2(x, a, b, c, d=0, e=0, f=0):
    assert d > 0
    assert e > 0
    assert f > 0
    time.sleep(np.random.rand() * 0.1)  # random delay
    return x**2

@pytest.mark.parametrize('map', [
    pytest.param(parallel_map, id='parallel_map'),
    pytest.param(loky_pmap, id='loky_pmap'),
    pytest.param(serial_map, id='serial_map'),
])
@pytest.mark.parametrize('num_cpus',
                         [1, 2],
                         ids=['1', '2'])
def test_map(map, num_cpus):
    if map is loky_pmap:
        loky = pytest.importorskip("loky")

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}
    map_kw = {
        'job_timeout': threading.TIMEOUT_MAX,
        'timeout': threading.TIMEOUT_MAX,
        'num_cpus': num_cpus,
    }

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    y2 = map(_func2, x, args, kwargs, map_kw=map_kw)
    assert ((np.array(y1) == np.array(y2)).all())


@pytest.mark.parametrize('map', [
    pytest.param(parallel_map, id='parallel_map'),
    pytest.param(loky_pmap, id='loky_pmap'),
    pytest.param(serial_map, id='serial_map'),
])
@pytest.mark.parametrize('num_cpus',
                         [1, 2],
                         ids=['1', '2'])
def test_map_accumulator(map, num_cpus):
    if map is loky_pmap:
        loky = pytest.importorskip("loky")
    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}
    map_kw = {
        'job_timeout': threading.TIMEOUT_MAX,
        'timeout': threading.TIMEOUT_MAX,
        'num_cpus': num_cpus,
    }
    y2 = []

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    map(_func2, x, args, kwargs, reduce_func=y2.append, map_kw=map_kw)
    assert ((np.array(sorted(y1)) == np.array(sorted(y2))).all())


class CustomException(Exception):
    pass


def func(i):
    if i % 2 == 1:
        raise CustomException(f"Error in subprocess {i}")
    return i


@pytest.mark.parametrize('map', [
    pytest.param(parallel_map, id='parallel_map'),
    pytest.param(loky_pmap, id='loky_pmap'),
    pytest.param(serial_map, id='serial_map'),
])
def test_map_pass_error(map):
    if map is loky_pmap:
        loky = pytest.importorskip("loky")

    with pytest.raises(CustomException) as err:
        map(func, range(10))
    assert "Error in subprocess" in str(err.value)


@pytest.mark.parametrize('map', [
    pytest.param(parallel_map, id='parallel_map'),
    pytest.param(loky_pmap, id='loky_pmap'),
    pytest.param(serial_map, id='serial_map'),
])
def test_map_store_error(map):
    if map is loky_pmap:
        loky = pytest.importorskip("loky")

    with pytest.raises(MapExceptions) as err:
        map(func, range(10), map_kw={"fail_fast": False})
    map_error = err.value
    assert "iterations failed" in str(map_error)
    for iter, error in map_error.errors.items():
        assert isinstance(error, CustomException)
        assert f"Error in subprocess {iter}" == str(error)
    for n, result in enumerate(map_error.results):
        if n % 2 == 0:
            # Passed
            assert result == n
        else:
            assert result is None
