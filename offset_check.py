"""
Estimate the average bias of a submission from the leaderboard, assuming an RMSE-type metric.
Submit the same predictions shifted by +delta (e.g. +0.10) and pass the three scores:

    python experiments/offset_check.py <score_of_original> <score_of_shifted> <delta>
    python experiments/offset_check.py 0.3494 0.3581 0.10      # -> bias about -0.019

Because  RMSE_shifted^2 = RMSE^2 + 2 * delta * bias + delta^2 .
A shift is only worth applying if |bias| is clearly larger than the noise of the public split (about 0.04).
"""
import sys
base, shifted, delta = map(float, sys.argv[1:4])
bias = (shifted ** 2 - base ** 2 - delta ** 2) / (2 * delta)
print(f"estimated bias (prediction - truth): {bias:+.4f}")
print(f"best constant shift: {-bias:+.3f} | expected score after it: {(base ** 2 - bias ** 2) ** 0.5:.4f}")
