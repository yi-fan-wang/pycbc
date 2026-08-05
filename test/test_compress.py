import numpy

from pycbc.waveform.compress import _select_refinement_indices


def test_refinement_skips_atomic_high_error_intervals():
    sample_index = numpy.array([0, 1, 2, 10])
    vecdiffs = numpy.array([3.0, 2.0, 1.0])

    selected = _select_refinement_indices(
        sample_index, vecdiffs, tolerance=0.5,
    )

    assert selected == [6]


def test_refinement_stops_only_when_all_intervals_are_atomic():
    sample_index = numpy.array([0, 1, 2, 3])
    vecdiffs = numpy.array([3.0, 2.0, 1.0])

    selected = _select_refinement_indices(
        sample_index, vecdiffs, tolerance=0.5,
    )

    assert selected == []
