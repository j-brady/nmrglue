import pytest
import numpy as np

from nmrglue.analysis.peakpick import pick
from nmrglue.analysis.linesh import sim_NDregion


def simulate_peaks(size, ls, params, amps):
    """Simulate an NMR spectrum with noise (crude)

    Parameters
    ----------
    size : Tuple(int,)
        size (shape) of spectrum in points
    ls : List[str]
        list of lineshape models ('l','g','pv')
    params : List[List[Tuple[float, float]]]
        parameters for peaks [[(center, fwhm)]]
    amps : List[float]
        amplitude of peaks

    Returns
    -------
    peaks : np.recarray
    """
    # simulate peaks
    peaks = sim_NDregion(size, ls, params, amps)
    # add some noise
    noise = np.random.normal(0, 0.1, size)
    peaks = peaks + noise 
    return peaks


@pytest.fixture
def x_axis_1():
    return 0


@pytest.fixture
def x_axis_2():
    return 1000


@pytest.fixture
def single_peak(x_axis_1):
    return simulate_peaks(size=[2**15,], ls=['l',], params=[[(x_axis_1,20)]], amps=[10,])


@pytest.fixture
def two_peaks(x_axis_1, x_axis_2):
    return simulate_peaks(size=[2**15,], ls=['l',], params=[[(x_axis_1,20)],[(x_axis_2,20)]], amps=[10,10,])


def test_pick(single_peak, x_axis_1):
    picked_peak = pick(single_peak, pthres=5)
    assert len(picked_peak) == 1
    assert abs(picked_peak["X_AXIS"][0] - x_axis_1) == 0.0


def test_two_peaks(two_peaks, x_axis_1, x_axis_2):
    picked_peaks = pick(two_peaks, pthres=5)
    assert len(picked_peaks) == 2
    assert abs(picked_peaks["X_AXIS"][1] - x_axis_2) == 0.0
    assert abs(picked_peaks["X_AXIS"][0] - x_axis_1) == 0.0