"""Discrete skills cut out of the continuous difficulty space.

`spacenav.demand` gives a continuum (D = eta / A) and `spacenav.opt.necessity`
gives the shape of a level's solution set.  Neither tells a player what they are
supposed to be learning.  This does: each lesson is a named skill with a
*measurable* signature, so a level is assigned to a lesson by what it demands,
not by which generator produced it.

The ladder is meant to be walked in order, and each lesson has a **challenge
branch** that pushes its own skill until it leaves human reach.  The branches are
not harder versions of the whole game; they are one skill taken to the point
where trained intuition stops being the right tool.  `one-exact-boost` is the
extreme case on purpose: a single irreplaceable commitment inside a window
narrower than a human reaction, which an agent should pass and a person should
not.

Level 0 deserves a note.  A level that solves itself when you touch nothing is
not a defective level, it is the opening lesson, and it works the way "hold
right" does in a platformer: the player does nothing, something complicated
happens, and what they learn is that the field is doing real work.  An earlier
version of this code treated a free ride as a generator bug and threw it away.
"""

from typing import Callable, NamedTuple

from spacenav.opt.probe import HUMAN

# Below this uncertainty exponent the win/lose boundary is fractal at any
# precision anyone can command: see `opt.necessity.boundary_exponent`.
#
# That on its own is not a defect.  A fractal region noises the flight and forces
# the pilot to adapt afterwards, which is exactly what a memorised open-loop plan
# cannot survive -- and closed-loop skill is the only kind that transfers to a
# moving opponent.  What decides whether it is a skill or a lottery is *where the
# chaos sits relative to the slack*: upstream of spare fuel and time, re-planning
# recovers it; at the moment of commitment with nothing left, it does not.
ALPHA_RESOLVABLE = 0.45


class Lesson(NamedTuple):
    key: str
    name: str
    teaches: str
    signature: str
    branch: str
    test: Callable[[dict], bool]


def _g(f, k, default=0.0):
    v = f.get(k, default)
    return default if v is None else v


# Ordered most specific first: a level is given the first lesson it matches, and
# `tags` records every lesson whose signature it satisfies.
LESSONS = (
    Lesson(
        "free-ride", "Gravity Is Cool",
        "The field does real work. Watch before you touch anything.",
        "coasting alone completes the tour",
        "more targets, and arcs that wind further before collecting them",
        lambda f: bool(_g(f, "coast_ok", False)),
    ),
    Lesson(
        "rolling-with-it", "Rolling With It",
        "Stop committing. Fly it, see where you end up, and fix it from there.",
        "a fractal boundary upstream of the slack: an open-loop plan loses it and "
        "re-planning recovers it",
        "move the chaos later, leaving less room to recover",
        lambda f: (_g(f, "alpha", 1.0) < ALPHA_RESOLVABLE
                   and bool(f.get("replanned", False)) and not bool(f.get("open_loop", True))),
    ),
    Lesson(
        "lottery", "Nobody's Skill",
        "Nothing, and that is the point: this one is not teachable.",
        f"fractal at any usable precision (alpha < {ALPHA_RESOLVABLE}) and "
        "re-planning does not rescue it either: the pilot can choose right and still lose",
        "not a branch -- reported so it can be kept off the path",
        lambda f: (f.get("alpha") is not None and _g(f, "alpha", 1.0) < ALPHA_RESOLVABLE
                   and _g(f, "sufficiency") > 0.0
                   and not bool(f.get("replanned", False))),
    ),
    Lesson(
        "one-exact-boost", "One Exact Boost",
        "Nothing. This is the branch where intuition is the wrong instrument.",
        "a single irreplaceable decision inside a window under a reaction time, "
        "on a boundary sharp enough to aim at",
        "narrow the window until only a machine can hit it",
        lambda f: (_g(f, "concentration") > 0.55 and _g(f, "n_critical") <= 2
                   and 0 < _g(f, "window_s", 1e9) < HUMAN.window_min
                   and _g(f, "alpha", 1.0) >= ALPHA_RESOLVABLE),
    ),
    Lesson(
        "gravity-assist", "All About Fuel",
        "Let the field accelerate you; the engine cannot afford the whole trip.",
        "the tank holds less delta-v than gravity delivers over the flight (A < 1)",
        "cut the tank further, until only an assist closes the trip",
        lambda f: _g(f, "A", 1e9) < 1.0,
    ),
    Lesson(
        "conics", "Thinking In Conics",
        "Straight-line aiming is wrong here. Think about the orbit you are on.",
        "a one-body model of the forces cannot fly it (D >= 0.1)",
        "raise D: contested placement in comparable-mass systems",
        lambda f: _g(f, "D") >= 0.1,
    ),
    Lesson(
        "leading", "Careful Leading",
        "Aim where the target will be, not where it is.",
        "burn time and burn heading are coupled across the winning set",
        "faster targets, and tighter arrival radii",
        lambda f: abs(_g(f, "leading")) > 0.35 and _g(f, "moving_frac") > 0.5,
    ),
    Lesson(
        "dead-heading", "Dead Heading",
        "Point at it and burn. The simplest thing that works.",
        "one burn wins, from a broad spread of times and directions",
        "shrink the spread until the burn has to be timed",
        lambda f: _g(f, "mdl", 99) <= 4 and _g(f, "sufficiency") > 0.04,
    ),
)

BY_KEY = {ls.key: ls for ls in LESSONS}
ORDER = tuple(ls.key for ls in LESSONS)
# the order a player should meet them, which is not the matching order above
PATH = ("free-ride", "dead-heading", "leading", "conics", "gravity-assist",
        "rolling-with-it")
BRANCHES = ("one-exact-boost",)
# matched and reported, but deliberately not taught
EXCLUDED = ("lottery",)

# The instrument each lesson earns.  These are the measured quantities of
# `spacenav.demand` and `spacenav.opt.necessity` turned into cockpit hardware:
# the player is given the abstraction exactly when the lesson needs it, and the
# last one exists because its lesson cannot be flown by feel at all.
INSTRUMENTS = {
    "free-ride": dict(name="none",
                      shows="nothing. Watch the field move the ship."),
    "dead-heading": dict(name="target marker",
                         shows="where the target is, and your heading."),
    "leading": dict(name="lead indicator",
                    shows="the intercept point, not the target's current position."),
    "conics": dict(name="tidal discriminator",
                   shows="eta, the share of local force that is not the dominant body: "
                         "where a one-body instinct is lying to you."),
    "gravity-assist": dict(name="budget gauge",
                           shows="A, the tank measured against the impulse gravity will "
                                 "deliver: below 1 you cannot power through."),
    "rolling-with-it": dict(name="predictability cone",
                            shows="an ensemble instead of a single predicted line. Where the "
                                  "cone opens, commit to nothing and keep fuel in hand."),
    "one-exact-boost": dict(name="flight computer",
                            shows="the viable set of (burn time, heading) for the state you "
                                  "are in. The lesson is unflyable by feel, so the "
                                  "instrument is what lets a person fly it at all."),
}

# Shown on every level once earned, because it is the honest answer to the
# chaos problem rather than an aid: the prediction ribbon fans into a cone at
# the point where forward prediction stops being deterministic.
PREDICTION_FAN = dict(
    name="predictability cone",
    shows="an ensemble of nearby trajectories instead of one line. Where the cone "
          "opens, the flight has become chaotic and no amount of precision will "
          "tell you which branch you are on.")


def tags(features: dict):
    """Every lesson whose signature this level satisfies."""
    return tuple(ls.key for ls in LESSONS if ls.test(features))


def classify(features: dict):
    """The lesson a level belongs to, or None when it teaches nothing nameable."""
    t = tags(features)
    return t[0] if t else None


def human_viable(features: dict, human=HUMAN):
    """Could a person be expected to fly this, given measured human limits?

    Two ways to fail: a window too narrow to aim at inside a reaction time, or a
    plan too fragile to survive one control tick of timing slop.
    """
    window = _g(features, "window_s", 0.0)
    jitter = features.get("jitter_1tick")
    if window and window < human.window_min:
        return False
    if jitter is not None and jitter < human.jitter_pass:
        return False
    return True


def resolvable(features: dict):
    """Is this level's win/lose boundary sharp enough to be worth aiming at?"""
    a = features.get("alpha")
    return True if a is None else float(a) >= ALPHA_RESOLVABLE


def describe(key):
    ls = BY_KEY[key]
    return dict(key=ls.key, name=ls.name, teaches=ls.teaches, signature=ls.signature,
                branch=ls.branch, on_path=key in PATH, is_branch=key in BRANCHES,
                excluded=key in EXCLUDED, instrument=INSTRUMENTS.get(key))


def syllabus():
    """The lessons in teaching order, the branches off them, and what is not taught."""
    return dict(path=[describe(k) for k in PATH],
                branches=[describe(k) for k in BRANCHES],
                excluded=[describe(k) for k in EXCLUDED],
                always=PREDICTION_FAN, alpha_resolvable=ALPHA_RESOLVABLE)
