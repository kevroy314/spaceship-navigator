"""The mission-spec grammar: `<stops>w<moving%>m[d][c][l]`.

Cheap, pure-Python coverage of the one string that ties the web client, the
curriculum, the episode ids and the pool files together.  The flags are
`d` docking, `c` contested placement, `l` low thrust, and they are positional.
"""

import pytest

from spacenav import constants as C
from spacenav import episodes as EP


@pytest.mark.parametrize("spec,stops,moving,dock,chaos,low", [
    ("1w0m", 1, 0, False, False, False),
    ("3w50m", 3, 50, False, False, False),
    ("1w100md", 1, 100, True, False, False),
    ("2w100mc", 2, 100, False, True, False),
    ("4w25ml", 4, 25, False, False, True),
    ("1w100mdcl", 1, 100, True, True, True),
    ("6w100mcl", 6, 100, False, True, True),
])
def test_spec_summary(spec, stops, moving, dock, chaos, low):
    s = EP.spec_summary(spec)
    assert s == dict(stops=stops, moving=moving, docking=dock, chaos=chaos, low_thrust=low)
    assert EP.normalise_spec(spec) == spec


@pytest.mark.parametrize("legacy,expect", [("f", "1w0m"), ("r", "1w100md"), ("t", "3w50m")])
def test_legacy_single_letters_still_resolve(legacy, expect):
    """Recorded runs and the parity test's episode ids use the old letters."""
    assert EP.normalise_spec(legacy) == expect
    assert EP.episode_id("val_seen", 3, 1, legacy) == f"val_seen-3-1-{expect}"
    assert EP.parse_id(f"val_seen-3-1-{legacy}") == ("val_seen", 3, 1, expect)


def test_max_waypoints_is_the_cap_not_the_regex():
    """MAX_WAYPOINTS is 6, so a 6-stop tour must round-trip; 9w must clamp rather
    than produce a task with more waypoints than the Task arrays hold."""
    assert C.MAX_WAYPOINTS == 6
    assert EP.normalise_spec("6w0m") == "6w0m"
    assert EP.spec_to_config("6w0m").waypoints == (6, 6)
    assert EP.normalise_spec("9w0m") == f"{C.MAX_WAYPOINTS}w0m"
    assert EP.spec_to_config("9w0m").waypoints == (C.MAX_WAYPOINTS, C.MAX_WAYPOINTS)


@pytest.mark.parametrize("bad", ["", "0w0m", "1w0", "w0m", "1w0mx", "1w1000m",
                                 "1w0mld", "1w0mcd", "10w0m"])
def test_bad_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        EP.normalise_spec(bad)
    with pytest.raises(ValueError):
        EP.spec_to_config(bad)


def test_flags_drive_the_task_config():
    plain = EP.spec_to_config("2w50m")
    assert plain.p_rendezvous == 0.0 and plain.p_contested == 0.0
    assert plain.p_station == 0.5
    assert EP.spec_to_config("2w50md").p_rendezvous == 1.0
    assert EP.spec_to_config("2w50mc").p_contested == 1.0

    # `l` is "low thrust": a weaker engine *and* shorter legs, because the floor
    # is set by the episode clock -- a weak engine on a long leg is infeasible,
    # not hard.
    low = EP.spec_to_config("2w50ml")
    assert low.accel_range == (0.5, 1.0)
    assert low.accel_range[1] < plain.accel_range[0]
    assert low.max_dist < plain.max_dist

    # human missions get a comfortable tank, not the training band
    assert plain.deltav_budget == (0.9, 1.45)
    assert plain.deltav_budget[0] > C.DELTAV_BUDGET_RANGE[0]


def test_single_stop_specs_get_longer_legs():
    """One stop means one leg, so it can be longer without overrunning the clock."""
    assert EP.spec_to_config("1w0m").max_dist > EP.spec_to_config("3w0m").max_dist
    assert EP.spec_to_config("1w0ml").max_dist > EP.spec_to_config("3w0ml").max_dist


def test_constants_dict_exports_max_waypoints():
    """The browser reads MAX_WAYPOINTS out of this, so it must not go missing."""
    d = EP.constants_dict()
    assert d["MAX_WAYPOINTS"] == C.MAX_WAYPOINTS
    assert d["MAX_EPISODE_TICKS"] == C.MAX_EPISODE_TICKS
    assert d["SUBSTEPS"] == C.SUBSTEPS
