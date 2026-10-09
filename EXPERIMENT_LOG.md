# Experiment log

Chronological record of what was tried on the Kaggle challenge `shl-hiring-assessment-2026`, including the ideas that did not work.

- **LB** = public leaderboard score (about 60% of the test set, lower is better).
- **CV** = out-of-fold RMSE of 5-fold cross-validation on the 732 normal training clips. CV and LB are on different scales (CV is about 0.15-0.2 higher), so only compare CV with CV and LB with LB.
- Scripts that reproduce the analyses are in [`experiments/`](experiments/); the final pipeline is `solution.py`.

## 1. Data analysis (`experiments/eda.py`)

| Finding | Consequence |
|---|---|
| 769 train / 216 test clips, about 60 s, 16 kHz mono, labels 0-5 in 0.5 steps | Few samples: use pretrained features and a regularised regressor |
| 37 clips are labelled 0, all with file number >= 5000, much louder with almost no silence; transcripts are loops like "the the the" | Non-speech recordings; **none in test** -> left out of training |
| 212 filenames occur in both train and test, but the audio differs (different md5) | Filename is not an id across splits; every key is `(split, filename)` |
| Whisper transcripts: median about 100 words, some repetition loops | Transcripts can be noisy |

## 2. Submissions and lessons

| # | What | CV | LB | Lesson |
|---|---|---|---|---|
| 1 | TF-IDF Ridge + gradient boosting on simple transcript/audio features | 0.82 | 0.6048 | Baseline |
| 2 | + RoBERTa embedding SVR | ~0.66 | 0.4740 | Pretrained embeddings help a lot |
| 3 | 3 text embeddings (RoBERTa-base/large, DeBERTa-v3-base) SVR, C=1, + 0.15 weight GBM on features | 0.605 | 0.3852 | C=3 overfits; C=1 better; TF-IDF text model was weak |
| 4 | + Whisper-small encoder embedding, weight 0.4 | 0.547 | 0.3687 | Audio adds new information |
| 5 | Audio SVR with C=3, weight 0.7 | 0.529 | 0.3881 | Training RMSE 0.18: overfit. **CV improved, LB got worse** |
| 6 | Mean of several audio layers, weight 0.5 | 0.527 | 0.3829 | Same pattern |
| 7 | Average of submissions 3, 4, 6 | - | 0.3712 | Predictions correlate 0.95-0.99, averaging similar models gives nothing |
| 8 | Whisper-small + medium audio layers, weight 0.6 | - | 0.3862 | More audio is not better |
| 9 | A submission built with Whisper-medium embeddings | best single audio model in CV | 0.4337 | The best CV model was the worst on the leaderboard |
| 10 | DeBERTa-v3-large text + Whisper-small audio, 70/30 | - | 0.3672 | Bigger text model: tiny gain |
| 11 | Same with WavLM-large audio instead of Whisper | - | 0.3626 | A different audio model helped slightly |

## 3. Why CV and the leaderboard disagreed (`experiments/adversarial_validation.py`)

A logistic regression that sees only embeddings can separate train clips from test clips: AUC 0.73-0.79 for text embeddings and 0.79-0.84 for Whisper-small audio layers (0.5 would mean no difference).
So train and test are distributed differently, audio more than text, and random cross-validation overrates audio features.
From here on the audio weight was decided on the leaderboard, and everything else (C, epsilon, layers) was kept fixed.

## 4. Rebuild and the final model (`experiments/variants_and_validation.py`)

After a notebook restart all cached features were lost and rebuilt with a resumable pipeline (now `solution.py`). Twelve candidates were compared (A = 0.7 text + 0.3 WavLM, C = 0.7 text + 0.15 Whisper + 0.15 WavLM; trained on all rows / on the 732 normal rows / importance-weighted; with and without the linear stretch):

| Variant | CV (732 rows) | LB |
|---|---|---|
| A, all 769 rows | 0.557 | not submitted (closest to the 0.3626 submission) |
| Importance weighting | 0.585-0.587 | not submitted (worse CV) |
| Normal rows only (+ stretch) | 0.549 -> 0.532 | - |
| **C, normal rows, stretch, audio weight 0.30** | 0.532 | 0.3538 |
| Same, audio weight 0.40 | 0.519 | 0.3499 |
| Same, 5 audio layers (adds middle layers), weight 0.40 | 0.514 | 0.3517 |
| **Same, audio weight 0.45 (final)** | 0.515 | **0.3494** |

Lessons: removing the non-speech clips and stretching the predictions (SVR predictions are shrunk toward the mean, slope about 1.1-1.2) gave the largest late gains; adding more layers did not help even though CV said it would.

## 5. Checks that were not applied

- **Constant offset** (`experiments/offset_check.py`): submitting the final predictions + 0.10 scored 0.3581, which implies a bias of only about -0.02, so no offset was added.
- **Per-domain standardisation** of embeddings (standardise train and test separately) moved the mean of the test predictions by about +0.14. Using the offset result above this would cost about 0.02, so it was not submitted.
- **Whisper-medium transcripts** instead of Whisper-small: no improvement.

## 6. What I would do next

Fine-tune DeBERTa on the transcripts (several seeds), add grammar-specific features (acceptability scores, ASR confidence, pause statistics), build a validation set that matches the test distribution, and do error analysis per score band.

## Process note

An AI assistant (Claude) was used for ideas and code drafts, as allowed by the challenge rules. All experiments were run by me, the results were interpreted by me, and the choices above are mine to explain.
