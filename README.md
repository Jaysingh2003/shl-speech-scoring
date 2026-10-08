# SHL Hiring Assessment 2026 - speech scoring

Private Kaggle challenge (`shl-hiring-assessment-2026`): predict a score from 0 to 5 (steps of 0.5) for each audio clip of about 60 seconds.
Data: 769 labelled training clips, 216 test clips. The leaderboard metric is an error (lower is better); the public leaderboard uses about 60% of the test set.

**Best public leaderboard score: 0.3494** (final submission, `submission_final.csv`).

## Approach

```
audio --> Whisper-small transcript --> RoBERTa-base / RoBERTa-large / DeBERTa-v3-base / DeBERTa-v3-large
                                       (mean-pooled embeddings) --> one SVR each --> average        = text branch   (weight 0.55)
audio --> Whisper-small encoder (last layer) and WavLM-large (last layer), mean-pooled over time
                                       --> one SVR each --> average                                  = audio branch  (weight 0.45)

prediction = 0.55 * text + 0.45 * audio
           -> linear "stretch" fitted on out-of-fold predictions (slope clipped to [1, 1.5])
           -> clip to [1, 5]
```

All SVRs are `StandardScaler -> SVR(C=1, epsilon=0.1)` with fixed hyper-parameters.

Key decisions
- **The given data is not modified.** The only data choice is which training rows the models fit on: the 37 clips whose file number is >= 5000 are left out.
  They are all labelled 0, are much louder with almost no silence (non-speech recordings), and the test set contains no such file.
- **Stretch:** SVR predictions are pulled towards the mean (out-of-fold slope of about 1.1-1.2), so a 2-parameter linear map learned on out-of-fold predictions is applied.
- **Audio weight** (0.3 -> 0.4 -> 0.45) was chosen with the public leaderboard, not with cross-validation (see below).

## Results (public leaderboard, lower is better)

| Step | Score |
|---|---|
| TF-IDF Ridge + gradient boosting on simple features | 0.6048 |
| 3 text embeddings + SVR (+ small GBM on transcript features) | 0.3852 |
| + Whisper-small audio embedding (audio weight 0.4) | 0.3687 |
| DeBERTa-v3-large text + Whisper-small audio, 70/30 | 0.3672 |
| same with WavLM-large audio, 70/30 | 0.3626 |
| train on the 732 normal clips, 2 audio models averaged, stretch (audio weight 0.3) | 0.3538 |
| audio weight 0.40 | 0.3499 |
| **audio weight 0.45 (final)** | **0.3494** |

## What did not help

- Heavier audio blends (weights 0.5-0.7) in the earlier setup: better cross-validation, worse leaderboard.
- A submission built with Whisper-medium embeddings scored 0.4337 on the leaderboard, although the same features looked best in cross-validation.
- Adding middle layers of Whisper-small / WavLM (5 layers instead of 2): better cross-validation, leaderboard 0.3517 (worse than 0.3499).
- Importance weighting of training rows by "test-likeness": worse cross-validation, not submitted.
- Whisper-medium transcripts instead of Whisper-small: no improvement.
- A constant offset: a +0.10 shift scored 0.3581, which implies the remaining bias is only about -0.02, so no offset is applied.

## Notes and limitations

- Train and test are distributed differently: a classifier separates them with an AUC of about 0.73-0.79 on the text embeddings and 0.79-0.84 on the Whisper-small audio layers.
  Plain random 5-fold cross-validation therefore overestimates the benefit of audio features, which is why audio-related choices were checked on the leaderboard.
- Because of that, the audio weight is tuned on the 60% public split and the final result on the other 40% may differ.
- Single run, fixed seeds for cross-validation, no speaker-grouped validation (speaker ids are not given).
- Transcription and embeddings run on GPU in fp16 / fp32 mixes, so a re-run can differ slightly.

## Reproduce

Kaggle notebook with GPU (T4) and Internet on, competition data attached:

```
pip install -r requirements.txt
python solution.py        # in a Kaggle notebook cell: !python solution.py
```

Stages: Whisper transcripts (~40 min) -> text embeddings (~10 min) -> audio embeddings (~30 min) -> final model (~1 min).
Each stage saves its output in `/kaggle/working` and is skipped when the file already exists. The result is `submission_final.csv`.
Set `AUDIO_WEIGHT = 0.40` in `solution.py` for the 0.3499 variant.
