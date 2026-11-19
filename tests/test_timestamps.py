"""Overview and inspector frame grids. Expected values are frozen, not imported."""

from main_agent.protocol import overview_timestamps, specialist_timestamps


def test_overview_grids():
    assert overview_timestamps(0.5) == [0.0]
    assert overview_timestamps(5.0) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert overview_timestamps(10.5) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert overview_timestamps(16.0) == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ]
    assert overview_timestamps(30.0) == [
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
    ]
    assert overview_timestamps(60.0) == [
        0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 31.0, 35.0, 39.0, 43.0, 47.0, 51.0, 55.0, 59.0,
    ]
    assert overview_timestamps(90.0) == [
        0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 47.0, 53.0, 59.0, 65.0, 71.0, 77.0, 83.0, 89.0,
    ]


def test_specialist_grids():
    assert specialist_timestamps(0.0, 4.0) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert specialist_timestamps(1.2, 8.7) == [
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
    ]
    assert specialist_timestamps(10.0, 10.2) == [10.0]
    assert specialist_timestamps(0.0, 0.4) == [0.0]
    capped = specialist_timestamps(0.0, 40.0)
    assert len(capped) == 72
    assert capped[0] == 0.0
    assert capped[-1] == 39.5
    assert capped[5] == 3.0
