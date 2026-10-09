"""
Exploratory data analysis for the SHL speech-scoring data (no models needed).
Run:  python experiments/eda.py
Findings used later: (1) 37 label-0 clips are loud non-speech recordings with file number >= 5000 and none are in test,
(2) 212 filenames exist in both train and test but the audio is different, so filename is not a unique id across splits.
"""
import os, hashlib
import numpy as np, pandas as pd, soundfile as sf

DATA = os.environ.get("SHL_DATA", "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final")
TR, TE = f"{DATA}/train", f"{DATA}/test"
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
print("train", train.shape, "| test", test.shape)
print(train.label.describe()); print(train.label.value_counts().sort_index())

# 1) duration / sample rate / channels
info = lambda folder, f: (lambda i: (i.duration, i.samplerate, i.channels))(sf.info(os.path.join(folder, f)))
train["dur"] = [info(TR, f)[0] for f in train.filename]
print("\nduration (sec):"); print(train.dur.describe())
print("sample rate / channels of first clip:", info(TR, train.filename[0])[1:])

# 2) same filename in train and test: is it the same audio?
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
common = sorted(set(train.filename) & set(test.filename))
same = sum(md5(os.path.join(TR, f)) == md5(os.path.join(TE, f)) for f in common)
print(f"\nfilenames present in both splits: {len(common)} | identical audio: {same}")

# 3) the label-0 clips
train["num"] = train.filename.str.extract(r"(\d+)")[0].astype(int)
test["num"] = test.filename.str.extract(r"(\d+)")[0].astype(int)
print("\nlabel by (file number >= 5000):"); print(train.groupby(train.num >= 5000).label.describe())
print("test clips with file number >= 5000:", int((test.num >= 5000).sum()))

def level(path):                                   # loudness and share of near-silent samples
    a, _ = sf.read(path); a = a.mean(1) if a.ndim > 1 else a
    return float(np.sqrt((a ** 2).mean())), float((np.abs(a) < 0.01).mean())
st = [level(os.path.join(TR, f)) for f in train.filename]
train["rms"], train["quiet"] = zip(*st)
print("\nloudness (rms) and silent share (quiet), label == 0 vs the rest:")
print(train.groupby(train.label == 0)[["rms", "quiet"]].mean())
