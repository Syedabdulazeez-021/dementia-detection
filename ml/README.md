# ML Training Pipeline (blink · gaze · voice · fusion)

This folder makes every channel ML-trainable and lets you learn the single
combined score from data. **No model is created from thin air** — each one is
trained on REAL labelled examples you provide.

> Honesty rule: train each modality on data where `true_label` is a **confirmed
> diagnosis**, and never merge different people across modalities. A real
> single combined model needs the **same people** measured on all three tests.

---

## The one combined score

The app already outputs **one overall score** from all three tests using a
weighted average (voice 0.40, blink 0.35, gaze 0.25, renormalised if a test is
skipped). `train_fusion.py` replaces those fixed weights with weights **learned
from data** — same single score, just data-driven instead of hand-picked.

---

## Scripts

| Script | What it does | Input |
|--------|--------------|-------|
| `train_models.py` | Train an RF for **one** modality | a labelled CSV for that modality |
| `train_fusion.py` | Learn the **single combined score** | a same-subject CSV with the 3 scores |

```bash
# activate your venv first
python ml/train_models.py --modality blink --data my_blink_labelled.csv
python ml/train_models.py --modality gaze  --data my_gaze_labelled.csv
python ml/train_models.py --modality voice --data my_voice_labelled.csv
python ml/train_fusion.py  --data my_multimodal_labelled.csv
```

CSV formats are shown in `templates/` (`blink_template.csv`, `gaze_template.csv`,
`fusion_template.csv`). The app's own session logs (`retrain_data.csv`,
`dementia_features_log.xlsx`) already capture features you can label and feed in.

Each trained model is saved as a `.pkl`. Once you have real models, ask me to
wire them into the app's scoring with a rule-based fallback (one safe backend
change, no UI change): if the model file is present it is used, otherwise the
current rule-based score is used.

---

## Where to get REAL labelled data

**Voice (the established route):** DementiaBank's **Pitt corpus** and the
**ADReSS / ADReSSo** sets are Cookie-Theft recordings labelled AD vs control.
Access is free but gated: join DementiaBank (TalkBank), with a faculty
supervisor for students; the data **cannot be redistributed**. Start at
`https://dementia.talkbank.org/` and the ADReSS challenge pages.

**Eye movement / gaze (and blink):** Large labelled sets exist in published
studies (e.g. eye-movement cohorts of 250+ AD/MCI/normal subjects across
fixation, smooth-pursuit, prosaccade and antisaccade tasks), but they are
**access-by-request from the study authors**, not open downloads. Email the
corresponding authors of the relevant papers (see `docs/REFERENCES.md`) and ask
about data-sharing under a data-use agreement.

**Collecting your own:** the most reproducible route is to record sessions with
this very app from consenting participants whose status is later confirmed,
fill in `true_label`, and train on those. Start with whatever you can label and
re-train as the dataset grows.

---

## What NOT to do

- Do **not** stitch voice from one dataset with gaze from another into a fake
  "multimodal" file — the rows would not be real people.
- Do **not** train on synthetic/random data and report its accuracy as a
  result. Synthetic data is only ever for checking that the code runs.
