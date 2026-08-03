"""crat_events.py — per-session event collector.

One instance per recording session. Everything it holds is **in memory only**:
it never writes a file and never reads one, so the existing design decision
that blink and gaze keep no cross-session logs is preserved.

The collector sits *beside* the existing counters — it never replaces them.
The counters remain the source of truth for every score; these events exist so
the figures can show *when* things happened, not just how many.

    events = SessionEvents()

    # blink detector, next to `self.micro_sleeps.append(...)`
    events.log_microsleep(closure_start_time, closure_duration)
    # blink detector, next to `self.partial_blinks.append(...)`
    events.log_partial_blink(dip_time)

    # gaze trial loop, next to `self.correct_count += 1`
    events.log_trial(target_side, response_side, v_peak)

    events.summary()   # -> counts that must agree with the on-screen counters

Times are in **seconds from the start of the recording**. Sides are ``-1`` for
left and ``+1`` for right.
"""

LEFT = -1
RIGHT = +1


def _side(value, allow_none=False):
    """Coerce a side to -1 / +1, accepting the string forms the app uses.

    With allow_none=True, ``None`` passes through unchanged. That is used for
    the *response* side of a trial the participant never answered (a timeout);
    a target side is always -1 or +1.
    """
    if value is None and allow_none:
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        if v in ("L", "LEFT", "-1"):
            return LEFT
        if v in ("R", "RIGHT", "+1", "1"):
            return RIGHT
        raise ValueError(f"unrecognised side: {value!r}")
    v = int(value)
    if v not in (LEFT, RIGHT):
        raise ValueError(f"side must be -1 or +1, got {v}")
    return v


class SessionEvents:
    """Per-session, in-memory event collector for the blink and gaze tests."""

    def __init__(self):
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        """Clear every event. Called wherever the session counters are zeroed."""
        self.microsleeps = []       # [{'t': s, 'duration': s}]
        self.partial_blinks = []    # [{'t': s}]
        self.trials = []            # [{'target', 'response', 'v_peak', 'correct'}]

    # -- blink -------------------------------------------------------------
    def log_microsleep(self, start_time, duration):
        """Record one closure longer than the micro-sleep threshold.

        Args:
            start_time: closure onset, seconds from the start of the recording
            duration:   closure length in seconds (already computed by the
                        detector in order to evaluate the > 0.5 s test)
        """
        self.microsleeps.append({"t": float(start_time),
                                 "duration": float(duration)})

    def log_partial_blink(self, dip_time):
        """Record one incomplete blink.

        Args:
            dip_time: centre of the dip, seconds from the start of the recording
        """
        self.partial_blinks.append({"t": float(dip_time)})

    # -- gaze --------------------------------------------------------------
    def log_trial(self, target_side, response_side, v_peak):
        """Record one completed gaze trial.

        Args:
            target_side:   -1 left / +1 right (strings 'LEFT'/'RIGHT' accepted)
            response_side: -1 left / +1 right, or ``None`` for a trial that
                           timed out with no classified response
            v_peak:        peak saccade speed for this trial, in the units the
                           detector already produces

        A ``None`` response is never correct and is flagged ``timeout``.
        """
        t = _side(target_side)
        r = _side(response_side, allow_none=True)
        self.trials.append({
            "target": t,
            "response": r,
            "v_peak": float(v_peak),
            "correct": r is not None and t == r,
            "timeout": r is None,
        })

    # -- read-out ----------------------------------------------------------
    @property
    def n_microsleeps(self):
        return len(self.microsleeps)

    @property
    def n_partial_blinks(self):
        return len(self.partial_blinks)

    @property
    def n_trials(self):
        return len(self.trials)

    @property
    def n_correct(self):
        return sum(1 for t in self.trials if t["correct"])

    @property
    def n_timeouts(self):
        return sum(1 for t in self.trials if t.get("timeout"))

    def accuracy(self):
        """A_gaze = n_correct / n * 100, or 0.0 with no trials."""
        if not self.trials:
            return 0.0
        return self.n_correct / float(len(self.trials)) * 100.0

    def v_peaks(self):
        """One peak saccade speed per trial, in trial order."""
        return [t["v_peak"] for t in self.trials]

    def summary(self):
        """Counts for cross-checking against the on-screen counters."""
        return {
            "microsleeps": self.n_microsleeps,
            "partial_blinks": self.n_partial_blinks,
            "trials": self.n_trials,
            "correct": self.n_correct,
            "timeouts": self.n_timeouts,
            "accuracy": self.accuracy(),
        }

    def __repr__(self):
        s = self.summary()
        return (f"SessionEvents(microsleeps={s['microsleeps']}, "
                f"partial_blinks={s['partial_blinks']}, "
                f"trials={s['trials']}, accuracy={s['accuracy']:.0f}%)")
