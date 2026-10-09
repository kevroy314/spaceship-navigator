"""`spacenav.lessons`: a level is assigned a lesson by what it measurably demands.

These are pure dictionary -> label tests: no simulation, so they are the cheapest
guard in the suite against the matching order or a predicate drifting.
"""

import pytest

from spacenav import lessons as L
from spacenav.opt.probe import HUMAN

A_OK = L.ALPHA_RESOLVABLE + 0.1      # a resolvable boundary
A_FRACTAL = L.ALPHA_RESOLVABLE - 0.1  # fractal at any usable precision


def feat(**kw):
    """Features of an unremarkable level, overridden by kw.

    Explicit rather than empty so each test changes exactly one thing: with an
    empty dict every `_g` default fires at once and the match is accidental.
    """
    base = dict(coast_ok=False, alpha=A_OK, replanned=False, open_loop=True,
                sufficiency=0.0, concentration=0.0, n_critical=99, window_s=1e9,
                A=1e9, D=0.0, eta=0.0, leading=0.0, moving_frac=0.0, mdl=99)
    base.update(kw)
    return base


# --- the path ---------------------------------------------------------------

def test_free_ride_wins_over_everything():
    """Level 0 is matched first on purpose: a free ride is the opening lesson,
    not a defective level, whatever else happens to be true of it."""
    f = feat(coast_ok=True, A=0.2, D=0.9, mdl=1, sufficiency=0.5)
    assert L.classify(f) == "free-ride"
    assert "free-ride" in L.tags(f)
    # the others still show up as tags; only the first is the assignment
    assert "gravity-assist" in L.tags(f) and "conics" in L.tags(f)


def test_dead_heading():
    f = feat(mdl=2, sufficiency=0.2)
    assert L.classify(f) == "dead-heading"


def test_leading_needs_both_a_correlation_and_moving_targets():
    assert L.classify(feat(leading=0.6, moving_frac=0.8)) == "leading"
    # a correlation with nothing moving is not leading
    assert L.classify(feat(leading=0.6, moving_frac=0.1)) is None
    # negative correlation counts: aiming behind is still coupling
    assert L.classify(feat(leading=-0.6, moving_frac=0.8)) == "leading"


def test_conics_on_demand():
    assert L.classify(feat(D=0.2)) == "conics"
    assert L.classify(feat(D=0.05)) is None


def test_fuel_and_model_are_separated_by_eta():
    """`gravity-assist` and `conics` are mutually exclusive, decided by eta.

    They used to be decided by match order: A < 1 came first, so any fuel-tight
    level was `gravity-assist` and `conics` could never appear on one -- which is
    most levels (docs/decisions/0020). Now the question is *which constraint
    binds*: a tight tank over a field one body explains is a fuel lesson, and a
    tight tank where the field itself is the problem is a conics lesson.

    Both cases keep D = eta / A consistent, so the features describe a level that
    could exist.
    """
    fuel = feat(A=0.5, eta=0.02, D=0.04)          # tight tank, simple field
    assert L.classify(fuel) == "gravity-assist"
    assert set(L.tags(fuel)) == {"gravity-assist"}

    model = feat(A=0.5, eta=0.45, D=0.90)         # tight tank, contested field
    assert L.classify(model) == "conics"
    assert set(L.tags(model)) == {"conics"}


def test_gravity_assist_needs_a_simple_field():
    """A < 1 alone is not the signature any more; a contested field overrides it."""
    assert L.classify(feat(A=0.5, eta=0.02, D=0.04)) == "gravity-assist"
    assert L.classify(feat(A=0.5, eta=0.50, D=1.00)) != "gravity-assist"


def test_rolling_with_it():
    f = feat(alpha=A_FRACTAL, replanned=True, open_loop=False)
    assert L.classify(f) == "rolling-with-it"


# --- the branch and the exclusion ------------------------------------------

def test_one_exact_boost_needs_a_sub_reaction_window_on_a_sharp_boundary():
    f = feat(concentration=0.8, n_critical=1, window_s=0.5 * HUMAN.window_min, alpha=A_OK)
    assert L.classify(f) == "one-exact-boost"
    # a window a person can aim at is not the branch
    assert L.classify(dict(f, window_s=2 * HUMAN.window_min)) is None
    # a zero window means "never measured", not "infinitely narrow"
    assert L.classify(dict(f, window_s=0.0)) is None
    # and a fractal boundary is not something to aim at: that is a lottery, not
    # a branch where a machine beats a hand
    assert L.classify(dict(f, alpha=A_FRACTAL, sufficiency=0.1)) == "lottery"


def test_lottery_is_fractal_and_unrescued_by_replanning():
    f = feat(alpha=A_FRACTAL, sufficiency=0.1, replanned=False)
    assert L.classify(f) == "lottery"
    # re-planning rescuing it makes it a skill instead
    assert L.classify(dict(f, replanned=True, open_loop=False)) == "rolling-with-it"
    # nothing ever wins: no winning set, so nothing to call a lottery
    assert "lottery" not in L.tags(dict(f, sufficiency=0.0))


@pytest.mark.parametrize("alpha", [None, 0.0])
def test_unmeasurable_alpha_is_never_a_lottery(alpha):
    """`opt.necessity.boundary_exponent` reports alpha=None when its winning set
    was too small to fit a slope to.  That is an absence of evidence, and calling
    it a lottery would condemn levels purely for being hard to measure.

    Note the asymmetry under test: alpha=0.0 *is* a measurement of a fractal
    boundary and must classify, while alpha=None must not.
    """
    f = feat(alpha=alpha, sufficiency=0.1, replanned=False)
    if alpha is None:
        assert "lottery" not in L.tags(f)
        assert L.classify(f) is None
        assert L.resolvable(f) is True      # unknown is not condemned
    else:
        assert L.classify(f) == "lottery"
        assert L.resolvable(f) is False


def test_unmeasurable_alpha_is_not_rolling_with_it_either():
    f = feat(alpha=None, replanned=True, open_loop=False)
    assert "rolling-with-it" not in L.tags(f)


# --- bookkeeping -----------------------------------------------------------

def test_classify_returns_none_for_a_level_that_teaches_nothing():
    assert L.classify(feat()) is None
    assert L.tags(feat()) == ()


def test_syllabus_is_consistent():
    keys = set(L.BY_KEY)
    assert set(L.PATH) | set(L.BRANCHES) | set(L.EXCLUDED) == keys
    assert not (set(L.PATH) & set(L.EXCLUDED))
    s = L.syllabus()
    assert [d["key"] for d in s["path"]] == list(L.PATH)
    assert all(d["on_path"] for d in s["path"])
    assert all(d["excluded"] for d in s["excluded"])
    # every taught lesson earns an instrument
    for k in L.PATH + L.BRANCHES:
        assert L.describe(k)["instrument"] is not None, k


def test_human_viable():
    assert L.human_viable(feat(window_s=1.0)) is True
    assert L.human_viable(feat(window_s=0.5 * HUMAN.window_min)) is False
    # an unmeasured window (0.0) does not disqualify a level
    assert L.human_viable(feat(window_s=0.0)) is True
    assert L.human_viable(dict(feat(), jitter_1tick=HUMAN.jitter_pass - 0.1)) is False
    assert L.human_viable(dict(feat(), jitter_1tick=None)) is True
