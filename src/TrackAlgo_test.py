from TrackAlgo import TrackAlgo
import numpy as np
import pytest
def test_one():
    testTrack = TrackAlgo(10, 30)
    testResults = testTrack.runAlgo()
    assert testResults[0] == pytest.approx(5 * np.sqrt(3))
    assert testResults[1] == pytest.approx(5)
