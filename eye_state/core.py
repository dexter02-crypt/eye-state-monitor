"""Calibrated eye geometry and time-based events. No camera or ML dependencies."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math
from statistics import median
from local_support import finite

STATES = ('OPEN', 'CLOSED', 'ASYMMETRIC', 'UNKNOWN')


def eye_aspect_ratio(points) -> float:
    if len(points) != 6 or any(len(p) != 2 for p in points):
        raise ValueError('EAR requires six two-dimensional points in contour order.')
    points = [tuple(finite(x, 'coordinate') for x in p) for p in points]
    width = math.dist(points[0], points[3])
    if width < 1e-8:
        raise ValueError('Degenerate eye width.')
    return (math.dist(points[1], points[5]) + math.dist(points[2], points[4])) / (2 * width)


def quantile(values, q):
    values = sorted(finite(v, 'sample') for v in values)
    if not values:
        raise ValueError('Empty samples.')
    position = (len(values)-1)*q
    a, b = math.floor(position), math.ceil(position)
    return values[a] + (values[b]-values[a])*(position-a)


@dataclass(frozen=True)
class Thresholds:
    close: float
    reopen: float
    def __post_init__(self):
        if not 0 <= finite(self.close, 'close') < finite(self.reopen, 'reopen') <= 1:
            raise ValueError('Require 0 <= close < reopen <= 1.')


def calibrate_eye(open_samples, closed_samples) -> Thresholds:
    if len(open_samples) < 20 or len(closed_samples) < 20:
        raise ValueError('At least 20 accepted samples are needed for each calibration phase.')
    if any(not 0 <= finite(v, 'EAR sample') <= 1 for v in [*open_samples, *closed_samples]):
        raise ValueError('Invalid EAR sample.')
    # Avoid thresholds at noisy extrema. These are engineering defaults, not clinical cutoffs.
    lower_open = quantile(open_samples, .10)
    upper_closed = quantile(closed_samples, .90)
    gap = lower_open - upper_closed
    if gap < .035 or median(open_samples) < .12:
        raise ValueError('Open/closed samples do not separate reliably. Improve framing/light and recalibrate.')
    return Thresholds(upper_closed + .35*gap, upper_closed + .65*gap)


def classify(value: float, previous: str, limits: Thresholds) -> str:
    value = finite(value, 'EAR')
    if not 0 <= value <= 1:
        return 'UNKNOWN'
    if value <= limits.close:
        return 'CLOSED'
    if value >= limits.reopen:
        return 'OPEN'
    return previous if previous in ('OPEN', 'CLOSED') else 'UNKNOWN'


@dataclass(frozen=True)
class Observation:
    time_s: float
    state: str
    reason: str
    left_ear: float | None
    right_ear: float | None
    closed_for_s: float
    blinks: int
    prolonged_events: int
    prolonged: bool
    known_time_s: float
    known_fraction: float
    closed_fraction_of_known: float | None


class EyeTracker:
    def __init__(self, left: Thresholds, right: Thresholds, *, prolonged_s=1.5,
                 min_blink_s=.07, max_blink_s=.6, max_gap_s=.25, window_s=30.0):
        for name, value in [('prolonged_s', prolonged_s), ('min_blink_s', min_blink_s),
                            ('max_blink_s', max_blink_s), ('max_gap_s', max_gap_s), ('window_s', window_s)]:
            if finite(value, name) <= 0:
                raise ValueError('Timing values must be positive.')
        if not min_blink_s < max_blink_s < prolonged_s or max_gap_s > window_s:
            raise ValueError('Inconsistent timing parameters.')
        self.left_limits, self.right_limits = left, right
        self.prolonged_s, self.min_blink_s, self.max_blink_s = prolonged_s, min_blink_s, max_blink_s
        self.max_gap_s, self.window_s = max_gap_s, window_s
        self.last = None
        self.first = None
        self.left = self.right = self.state = 'UNKNOWN'
        self.closed_start = None
        self.preceded_by_open = False
        self.announced = False
        self.blinks = self.prolonged_events = 0
        self.intervals = deque()

    def update(self, t: float, left: float | None, right: float | None, reason: str = '') -> Observation:
        t = finite(t, 'timestamp')
        if t < 0 or (self.last is not None and t <= self.last):
            raise ValueError('Timestamps must be non-negative and strictly increasing.')
        if self.first is None:
            self.first = t
        dt = 0 if self.last is None else t-self.last
        old_state = self.state
        gap = dt > self.max_gap_s
        valid = not reason and left is not None and right is not None
        if valid:
            try:
                valid = 0 <= finite(left, 'left EAR') <= 1 and 0 <= finite(right, 'right EAR') <= 1
            except ValueError:
                valid = False
        if gap:
            self.left = self.right = self.state = 'UNKNOWN'
            self.closed_start = None
            self.preceded_by_open = False
            self.announced = False
            old_state = 'UNKNOWN'
        if not valid:
            self.left = self.right = 'UNKNOWN'
            new_state = 'UNKNOWN'
            reason = reason or 'INVALID_MEASUREMENT'
            left = right = None
        else:
            self.left = classify(left, self.left, self.left_limits)
            self.right = classify(right, self.right, self.right_limits)
            new_state = self.left if self.left == self.right else 'ASYMMETRIC'
            if 'UNKNOWN' in (self.left, self.right):
                new_state = 'UNKNOWN'
            reason = 'SAMPLING_GAP_RESET' if gap else ('AMBIGUOUS_OR_ASYMMETRIC' if new_state not in ('OPEN','CLOSED') else '')
        # Conservatively count only intervals whose endpoints agree and are observed.
        # Transitions, asymmetric frames and missing samples add no known time.
        if dt > 0:
            label = new_state if not gap and new_state == old_state and new_state in ('OPEN','CLOSED') else 'UNKNOWN'
            self.intervals.append((self.last, t, label))
        if new_state == 'CLOSED':
            if self.closed_start is None:
                self.closed_start = t
                self.preceded_by_open = old_state == 'OPEN'
                self.announced = False
            duration = t - self.closed_start
            if duration >= self.prolonged_s and not self.announced:
                self.prolonged_events += 1
                self.announced = True
        else:
            if new_state == 'OPEN' and old_state == 'CLOSED' and self.closed_start is not None:
                duration = t-self.closed_start
                if self.preceded_by_open and self.min_blink_s <= duration <= self.max_blink_s:
                    self.blinks += 1
            self.closed_start = None
            self.preceded_by_open = False
            self.announced = False
            duration = 0.0
        self.state, self.last = new_state, t
        start = max(self.first, t-self.window_s)
        while self.intervals and self.intervals[0][1] <= start:
            self.intervals.popleft()
        known = closed = 0.0
        for a, b, label in self.intervals:
            seconds = max(0.0, b-max(a, start))
            if label in ('OPEN', 'CLOSED'):
                known += seconds
            if label == 'CLOSED':
                closed += seconds
        total = t-start
        return Observation(t, new_state, reason, left, right, duration,
                           self.blinks, self.prolonged_events, new_state=='CLOSED' and self.announced,
                           known, known/total if total else 0, closed/known if known else None)


def evaluate(rows):
    """Per-frame CLOSED-positive metrics. Unknowns remain visible, never count as correct."""
    tp = fp = tn = fn = unknown = closed_truth = open_truth = 0
    for row in rows:
        truth, prediction = row['truth'], row['prediction']
        if truth not in ('OPEN', 'CLOSED') or prediction not in STATES:
            raise ValueError('Truth must be OPEN/CLOSED; prediction must be a documented state.')
        closed_truth += truth == 'CLOSED'
        open_truth += truth == 'OPEN'
        if prediction in ('UNKNOWN', 'ASYMMETRIC'):
            unknown += 1
            fn += truth == 'CLOSED'
            continue
        tp += truth == 'CLOSED' and prediction == 'CLOSED'
        fp += truth == 'OPEN' and prediction == 'CLOSED'
        tn += truth == 'OPEN' and prediction == 'OPEN'
        fn += truth == 'CLOSED' and prediction == 'OPEN'
    n = closed_truth+open_truth
    if not n:
        raise ValueError('No evaluation rows.')
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/closed_truth if closed_truth else None
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return {'rows': n, 'closed_truth': closed_truth, 'open_truth': open_truth,
            'true_positive': tp, 'false_positive': fp, 'true_negative': tn,
            'closed_missed_including_abstentions': fn, 'abstentions': unknown,
            'coverage': (n-unknown)/n, 'closed_precision': precision,
            'closed_recall_including_abstentions': recall, 'closed_f1': f1,
            'overall_correct_fraction': (tp+tn)/n,
            'note': 'Per-frame eye-state metrics, not sleep classification or blink-event metrics.'}
