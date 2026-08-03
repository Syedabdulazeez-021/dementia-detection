"""crat_figures.py — shared CRAT figure style, blink figure, gaze figure.

One channel, one hue. Every figure in the application draws its colours from
the tables below and from nowhere else.

    Element          Dark       Mid        Light
    Blink            #0F766E    #14B8A6    #CCFBF1
    Gaze             #C2410C    #F97316    #FFEDD5
    Voice            #6D28D9    #8B5CF6    #EDE9FE
    Fusion / output  #0F172A    #1E293B    #E2E8F0

Reserved meanings, identical in every figure:
    normal-range band           BAND at BAND_ALPHA
    out-of-range / flagged      BAD
    in-range value              the channel's dark colour
    threshold line              dashed, 1.4 pt, channel dark colour

``apply_style()`` is called on import and sets the shared typography
process-wide, so importing this module is enough to restyle any figure drawn
afterwards.

Public API
    PAL, BAND, BAD, BAND_ALPHA, THRESH_LW
    apply_style()
    blink_figure(events, metrics=..., out_path=...)   -> path
    gaze_figure(events, gaze=..., out_path=...)       -> path
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Palette — the single source of truth for colour in this application
# ---------------------------------------------------------------------------
PAL = {
    "blink":  {"dark": "#0F766E", "mid": "#14B8A6", "light": "#CCFBF1"},
    "gaze":   {"dark": "#C2410C", "mid": "#F97316", "light": "#FFEDD5"},
    "voice":  {"dark": "#6D28D9", "mid": "#8B5CF6", "light": "#EDE9FE"},
    "fusion": {"dark": "#0F172A", "mid": "#1E293B", "light": "#E2E8F0"},
}

BAND = "#DCFCE7"        # normal-range band
BAND_ALPHA = 0.35
BAD = "#DC2626"         # out-of-range value or flagged event
THRESH_LW = 1.4         # threshold line width, in points

# Every hex a figure is allowed to use.
ALLOWED_HEX = {c.upper() for hue in PAL.values() for c in hue.values()}
ALLOWED_HEX.update({BAND.upper(), BAD.upper()})

# Anchors reused by the figures. These mirror the values already in the
# scoring / detector code; they are drawn, never applied to a score.
MICROSLEEP_S = 0.5              # dementia_analyzer micro-sleep closure test
GAZE_RT_HEALTHY_S = 0.5         # scoring.GAZE_RT_HEALTHY_S
GAZE_SPEED_HEALTHY = 300.0      # scoring.GAZE_SPEED_HEALTHY
BLINK_INTERVAL_LO = 2.0         # normal inter-blink interval band
BLINK_INTERVAL_HI = 6.0

_PARTIAL_MARK_H = 0.14          # fixed low height for partial-blink markers


# ---------------------------------------------------------------------------
# Shared typography / style
# ---------------------------------------------------------------------------
def apply_style():
    """Set the shared CRAT typography and axis style process-wide."""
    ink = PAL["fusion"]["dark"]
    muted = PAL["fusion"]["mid"]
    rule = PAL["fusion"]["light"]

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": ink,
        "axes.labelsize": 8.5,
        "axes.labelcolor": muted,
        "axes.edgecolor": rule,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": rule,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "text.color": ink,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": rule,
        "figure.facecolor": "white",
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


apply_style()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tidy(ax, channel=None):
    """Drop the top/right spines and tint the remaining ones."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["fusion"]["light"])


def _threshold(ax, y, channel, label=None, axis="h"):
    """Draw a threshold line: dashed, THRESH_LW pt, channel dark colour."""
    c = PAL[channel]["dark"]
    fn = ax.axhline if axis == "h" else ax.axvline
    fn(y, color=c, ls="--", lw=THRESH_LW, zorder=4, label=label)


def _band(ax, lo, hi, axis="h"):
    """Shade a normal-range band with the reserved colour and alpha."""
    fn = ax.axhspan if axis == "h" else ax.axvspan
    return fn(lo, hi, color=BAND, alpha=BAND_ALPHA, zorder=0)


def _empty(ax, message):
    """Annotate an axis that has no events, rather than dropping the panel."""
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color=PAL["fusion"]["mid"])


def ear_to_percent(ear):
    """EAR -> 0-100 % openness. Same mapping the GUI already uses."""
    return max(0.0, min(100.0, ((ear - 0.15) / (0.38 - 0.15)) * 100.0))


def _panel_tag(ax, tag):
    ax.text(-0.075, 1.06, tag, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=PAL["fusion"]["dark"], ha="left", va="bottom")


# ---------------------------------------------------------------------------
# Blink figure — three panels, teal
# ---------------------------------------------------------------------------
def blink_figure(events, metrics=None, out_path="fig_blink.png", dpi=300,
                 duration_s=None, title="Blink channel"):
    """Three-panel blink figure.

    (a) eye openness over time with the adaptive blink threshold
    (b) inter-blink intervals against the normal 2-6 s band
    (c) micro-sleep and partial-blink events across the recording

    Args:
        events:   a ``crat_events.SessionEvents`` (panel c)
        metrics:  the analyser metrics dict (``ear_history``, ``blink_times``,
                  ``current_threshold``); panels a and b are annotated as empty
                  without it
        out_path: PNG to write
        dpi:      output resolution
        duration_s: recording length for the panel-(c) x-axis; inferred when None

    Returns:
        the path written
    """
    metrics = metrics or {}
    C = PAL["blink"]
    ear_history = list(metrics.get("ear_history", []) or [])
    blink_times = list(metrics.get("blink_times", []) or [])

    # Recording-relative time base
    t0 = None
    if ear_history:
        t0 = ear_history[0].get("timestamp")
    elif blink_times:
        t0 = blink_times[0]

    if duration_s is None:
        if ear_history and t0 is not None:
            duration_s = max(1.0, float(ear_history[-1]["timestamp"]) - float(t0))
        else:
            ev_t = ([e["t"] + e["duration"] for e in events.microsleeps]
                    + [e["t"] for e in events.partial_blinks])
            duration_s = max(60.0, max(ev_t) * 1.05) if ev_t else 60.0

    fig = Figure(figsize=(8.0, 9.6), dpi=dpi)
    fig.suptitle(title, y=0.975, color=PAL["fusion"]["dark"])
    fig.subplots_adjust(left=0.11, right=0.95, top=0.925, bottom=0.07, hspace=0.42)

    # -- (a) eye openness over time ---------------------------------------
    ax1 = fig.add_subplot(3, 1, 1)
    if ear_history:
        times = [float(e["timestamp"]) - float(t0) for e in ear_history]
        left = [ear_to_percent(e["left"]) for e in ear_history]
        right = [ear_to_percent(e["right"]) for e in ear_history]
        ax1.plot(times, left, color=C["dark"], lw=0.9, alpha=0.9, label="Left eye")
        ax1.plot(times, right, color=C["mid"], lw=0.9, alpha=0.9, label="Right eye")
        thr = ear_to_percent(metrics.get("current_threshold", 0.25))
        _threshold(ax1, thr, "blink", label=f"Blink threshold ({thr:.0f} %)")
        ax1.set_ylim(0, 100)
        ax1.set_xlim(0, duration_s)
        ax1.legend(loc="lower right", ncol=3)
    else:
        _empty(ax1, "Blink test not run")
    ax1.set_title("(a)  Eye openness over time")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("eye openness (%)")
    _tidy(ax1)
    _panel_tag(ax1, "")

    # -- (b) inter-blink intervals ----------------------------------------
    ax2 = fig.add_subplot(3, 1, 2)
    if len(blink_times) > 1:
        intervals = np.diff(np.asarray(blink_times, dtype=float))
        x = np.arange(1, len(intervals) + 1)
        colors = [C["dark"] if BLINK_INTERVAL_LO <= iv <= BLINK_INTERVAL_HI else BAD
                  for iv in intervals]
        _band(ax2, BLINK_INTERVAL_LO, BLINK_INTERVAL_HI)
        ax2.bar(x, intervals, color=colors, width=0.62, zorder=3)
        ax2.set_ylim(0, max(BLINK_INTERVAL_HI + 1.0, float(intervals.max()) * 1.2))
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{i}" for i in x])
        n_out = sum(1 for iv in intervals if not BLINK_INTERVAL_LO <= iv <= BLINK_INTERVAL_HI)
        ax2.legend(handles=[
            Patch(facecolor=BAND, alpha=BAND_ALPHA,
                  label=f"normal {BLINK_INTERVAL_LO:g}–{BLINK_INTERVAL_HI:g} s"),
            Patch(facecolor=C["dark"], label=f"in range ({len(intervals) - n_out})"),
            Patch(facecolor=BAD, label=f"out of range ({n_out})"),
        ], loc="upper right", ncol=3)
    else:
        _empty(ax2, "Not enough blinks for an interval")
    ax2.set_title("(b)  Blink regularity — interval between consecutive blinks")
    ax2.set_xlabel("interval #")
    ax2.set_ylabel("interval (s)")
    _tidy(ax2)

    # -- (c) micro-sleep + partial-blink events ---------------------------
    ax3 = fig.add_subplot(3, 1, 3)
    ms = list(events.microsleeps)
    pb = list(events.partial_blinks)

    if ms:
        for e in ms:
            ax3.bar(e["t"] + e["duration"] / 2.0, e["duration"],
                    width=max(e["duration"], 0.30), color=BAD, zorder=3)
    if pb:
        ax3.plot([e["t"] for e in pb], [_PARTIAL_MARK_H] * len(pb),
                 linestyle="none", marker="v", ms=4.5, color=C["mid"], zorder=4)

    _threshold(ax3, MICROSLEEP_S, "blink")
    top = max(MICROSLEEP_S * 1.6,
              max((e["duration"] for e in ms), default=0.0) * 1.25)
    ax3.set_ylim(0, top)
    ax3.set_xlim(0, duration_s)
    ax3.text(duration_s * 0.995, MICROSLEEP_S, f" {MICROSLEEP_S:g} s ", ha="right",
             va="bottom", fontsize=6.5, color=C["dark"])

    ax3.legend(handles=[
        Patch(facecolor=BAD, label=f"micro-sleep ({len(ms)})"),
        Line2D([0], [0], linestyle="none", marker="v", ms=4.5, color=C["mid"],
               label=f"partial blink ({len(pb)})"),
        Line2D([0], [0], ls="--", lw=THRESH_LW, color=C["dark"],
               label=f"{MICROSLEEP_S:g} s closure threshold"),
    ], loc="upper right", ncol=3)

    # Zero of either type is annotated, never dropped.
    if not ms and not pb:
        _empty(ax3, "No micro-sleeps and no partial blinks in this session")
    elif not ms:
        ax3.text(0.02, 0.88, "0 micro-sleeps", transform=ax3.transAxes,
                 fontsize=8, color=PAL["fusion"]["mid"])
    elif not pb:
        ax3.text(0.02, 0.88, "0 partial blinks", transform=ax3.transAxes,
                 fontsize=8, color=PAL["fusion"]["mid"])

    ax3.set_title("(c)  Micro-sleeps and partial blinks over the recording")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("closure duration (s)")
    _tidy(ax3)

    fig.savefig(out_path, dpi=dpi)
    return out_path


# ---------------------------------------------------------------------------
# Gaze figure — three panels, orange
# ---------------------------------------------------------------------------
def gaze_figure(events, gaze=None, out_path="fig_gaze.png", dpi=300,
                title="Gaze channel"):
    """Three-panel gaze figure.

    (a) reaction time per trial against the 0.5 s normal latency
    (b) peak saccade speed per trial — exactly one bar per trial
    (c) target side vs participant response, per trial

    Args:
        events:   a ``crat_events.SessionEvents``; panels b and c come from it
        gaze:     the gaze results dict (``reaction_times``) for panel a
        out_path: PNG to write
        dpi:      output resolution

    Returns:
        the path written
    """
    gaze = gaze or {}
    C = PAL["gaze"]
    rt = list(gaze.get("reaction_times", []) or [])
    trials = list(events.trials)
    n = len(trials)

    fig = Figure(figsize=(8.0, 9.6), dpi=dpi)
    fig.suptitle(title, y=0.975, color=PAL["fusion"]["dark"])
    fig.subplots_adjust(left=0.11, right=0.95, top=0.925, bottom=0.07, hspace=0.42)

    # -- (a) reaction time per trial --------------------------------------
    ax1 = fig.add_subplot(3, 1, 1)
    if rt:
        x = np.arange(1, len(rt) + 1)
        _band(ax1, 0, GAZE_RT_HEALTHY_S)
        ax1.plot(x, rt, "-", color=C["mid"], lw=1.5, zorder=2)
        colors = [C["dark"] if v <= GAZE_RT_HEALTHY_S else BAD for v in rt]
        ax1.scatter(x, rt, c=colors, s=34, zorder=5)
        _threshold(ax1, GAZE_RT_HEALTHY_S, "gaze")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"T{i}" for i in x])
        ax1.set_ylim(0, max(1.0, max(rt) * 1.25))
        n_slow = sum(1 for v in rt if v > GAZE_RT_HEALTHY_S)
        ax1.legend(handles=[
            Patch(facecolor=BAND, alpha=BAND_ALPHA,
                  label=f"normal ≤ {GAZE_RT_HEALTHY_S:g} s"),
            Line2D([0], [0], ls="none", marker="o", color=C["dark"],
                   label=f"in range ({len(rt) - n_slow})"),
            Line2D([0], [0], ls="none", marker="o", color=BAD,
                   label=f"slow ({n_slow})"),
        ], loc="upper right", ncol=3)
        ax1.text(0.01, 0.93, f"mean {float(np.mean(rt)):.2f} s",
                 transform=ax1.transAxes, fontsize=7.5, color=PAL["fusion"]["mid"])
    else:
        _empty(ax1, "Gaze test not run")
    ax1.set_title("(a)  Reaction time per trial")
    ax1.set_xlabel("trial")
    ax1.set_ylabel("reaction time (s)")
    _tidy(ax1)

    # -- (b) peak saccade speed per trial ---------------------------------
    # One bar per trial. Not per-frame samples: exactly n bars for n trials.
    ax2 = fig.add_subplot(3, 1, 2)
    if n:
        vp = events.v_peaks()
        x = np.arange(1, n + 1)
        colors = [C["dark"] if v >= GAZE_SPEED_HEALTHY else BAD for v in vp]
        ax2.bar(x, vp, color=colors, width=0.62, zorder=3)
        _threshold(ax2, GAZE_SPEED_HEALTHY, "gaze")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"T{i}" for i in x])
        top = max(GAZE_SPEED_HEALTHY * 1.15, max(vp) * 1.3) if vp else GAZE_SPEED_HEALTHY
        ax2.set_ylim(0, top)
        n_slow = sum(1 for v in vp if v < GAZE_SPEED_HEALTHY)
        ax2.legend(handles=[
            Line2D([0], [0], ls="--", lw=THRESH_LW, color=C["dark"],
                   label=f"healthy ≥ {GAZE_SPEED_HEALTHY:g} px/s"),
            Patch(facecolor=C["dark"], label=f"in range ({n - n_slow})"),
            Patch(facecolor=BAD, label=f"below anchor ({n_slow})"),
        ], loc="upper right", ncol=3)
        ax2.text(0.01, 0.93, f"{n} trials, {n} V_peak values",
                 transform=ax2.transAxes, fontsize=7.5, color=PAL["fusion"]["mid"])
    else:
        _empty(ax2, "Gaze test not run")
    ax2.set_title("(b)  Peak saccade speed per trial (V_peak)")
    ax2.set_xlabel("trial")
    ax2.set_ylabel("V_peak (px/s)")
    _tidy(ax2)

    # -- (c) target vs response -------------------------------------------
    ax3 = fig.add_subplot(3, 1, 3)
    if n:
        x = np.arange(1, n + 1)
        # A timed-out trial has no response side; it is drawn as a red x on the
        # midline with a connector down from the target it never reached.
        answered = [(i, t) for i, t in enumerate(trials, start=1)
                    if t.get("response") is not None]
        timed_out = [(i, t) for i, t in enumerate(trials, start=1)
                     if t.get("response") is None]

        for i, tr in answered:
            if not tr["correct"]:
                ax3.plot([i, i], [tr["target"], tr["response"]],
                         color=BAD, lw=1.2, zorder=2)
        for i, tr in timed_out:
            ax3.plot([i, i], [tr["target"], 0.0], color=BAD, lw=1.2,
                     ls=":", zorder=2)

        # Target: hollow grey marker
        ax3.scatter(x, [t["target"] for t in trials], s=64,
                    facecolors="none", edgecolors=PAL["fusion"]["mid"],
                    linewidths=1.1, zorder=3, label="target side")
        # Response: filled, gaze-dark when correct, red when not
        if answered:
            ax3.scatter([i for i, _ in answered],
                        [t["response"] for _, t in answered], s=38,
                        c=[C["dark"] if t["correct"] else BAD for _, t in answered],
                        zorder=4, label="response")
        if timed_out:
            ax3.scatter([i for i, _ in timed_out], [0.0] * len(timed_out),
                        s=70, c=BAD, marker="x", linewidths=1.8, zorder=5,
                        label="no response")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"T{i}" for i in x])
        ax3.set_yticks([-1, 1])
        ax3.set_yticklabels(["left (−1)", "right (+1)"])
        ax3.set_ylim(-1.9, 1.9)
        ax3.set_xlim(0.4, n + 0.6)

        n_correct = events.n_correct
        acc = events.accuracy()
        n_timeout = len(timed_out)
        handles = [
            Line2D([0], [0], ls="none", marker="o", ms=7, markerfacecolor="none",
                   markeredgecolor=PAL["fusion"]["mid"], label="target"),
            Line2D([0], [0], ls="none", marker="o", color=C["dark"],
                   label=f"correct ({n_correct})"),
            Line2D([0], [0], ls="none", marker="o", color=BAD,
                   label=f"incorrect ({n - n_correct - n_timeout})"),
        ]
        if n_timeout:
            handles.append(Line2D([0], [0], ls="none", marker="x", color=BAD,
                                  label=f"no response ({n_timeout})"))
        ax3.legend(handles=handles, loc="upper right", ncol=len(handles))
        ax3.text(0.01, 0.93,
                 f"A_gaze = {n_correct}/{n} × 100 = {acc:.0f} %",
                 transform=ax3.transAxes, fontsize=8, fontweight="bold",
                 color=PAL["fusion"]["dark"])
    else:
        _empty(ax3, "Gaze test not run")
    ax3.set_title("(c)  Target side vs participant response")
    ax3.set_xlabel("trial")
    ax3.set_ylabel("side")
    _tidy(ax3)

    fig.savefig(out_path, dpi=dpi)
    return out_path
