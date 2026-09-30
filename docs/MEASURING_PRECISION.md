# Measure before claiming precision

No real-person accuracy score is supplied. Unit-test counts, synthetic traces, and a successful blank-image model check are not substitutes.

## A bounded, reproducible local experiment

Use your own consenting, seated webcam recording, not a vehicle or safety-critical situation. Keep calibration samples separate from evaluation clips. Keep thresholds fixed after calibration. Record the camera, resolution, FPS, model receipt SHA, code revision, lighting, glasses condition and approximate distance. Test normal open eyes, natural blinks, intentional long closures, head turns, looking down, occlusion, and leaving the frame. You are labelling visible eye state—not whether the subject is asleep.

Manually label a held-out clip before viewing the classifier outputs. Define what OPEN and CLOSED mean visually and decide how to handle genuinely unjudgeable source frames in advance. Record those exclusions and their count; do not retrospectively remove difficult examples to improve a score. Have a second person check ambiguous labels where possible. Several clips and conditions are more informative than thousands of nearly identical frames from one short clip. Performance for one person does not establish performance for other people.

Use `app.py video your_clip.mp4 --no-display --csv outputs/predictions.csv` with the chosen calibration. Frame times are frame index / FPS; use constant-frame-rate footage. Align manual labels by timestamp and put the independent truth plus recorded `state` in a CSV:

```csv
truth,prediction
OPEN,OPEN
CLOSED,CLOSED
CLOSED,UNKNOWN
OPEN,CLOSED
```

Run `app.py evaluate labels.csv --output outputs/evaluation.json`. The two required columns are exactly `truth,prediction`. Truth is OPEN/CLOSED. Prediction may also be UNKNOWN or ASYMMETRIC. Exporting live metrics does not magically provide truth labels.

## Read all the metrics together

CLOSED precision = correct CLOSED predictions / all CLOSED predictions. CLOSED recall = correct CLOSED predictions / all truly CLOSED labelled frames, including those for which the system abstains. Coverage = fraction of rows with a definite OPEN or CLOSED prediction. Overall correct fraction uses every evaluation row, so abstentions are not counted as correct. Undefined precision is null, not 100%.

The CSV tool's coverage is frame-count coverage; the live dashboard's coverage is time-weighted and conservatively excludes transition intervals. They are intentionally different. A system can obtain high precision by rarely making a prediction; report coverage and recall next to precision, not in a footnote. Also report per-condition results and the number of labelled frames, clips and people. Do not imply independent samples when adjacent video frames are correlated.

For blink/prolonged-event assessment, separately annotate event start/end times and establish an event-matching tolerance before evaluating. Count missed events and false events per minute; compare durations. The included evaluation CLI is per-frame and DOES NOT calculate those event-level statistics. Sleep classification, clinical conclusions and driving safety are outside scope.

## Included example is manufactured

`examples/synthetic-labels.csv` contains ten manually manufactured rows: one false positive, one false negative, and two abstentions. Its deliberately imperfect metrics only exercise the calculator. Never use its numbers as a project accuracy claim. `examples/labels-template.csv` contains only the header for your own labels.

Suggested README result statement after a genuine experiment:

“On [N] independently labelled frames from [C] held-out clips of [P] consenting people under [conditions], CLOSED precision was [value], recall [value], and prediction coverage [value]. The detector failed on [cases]. Calibration was separate. These results do not evaluate sleep or driver safety.”
