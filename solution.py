"""
SHL Hiring Assessment 2026 (Kaggle) - final solution

Pipeline:
  audio -> Whisper-small transcript -> 4 text embeddings (RoBERTa-base/large, DeBERTa-v3-base/large)
        -> SVR per embedding -> average                                   (text branch,  weight 0.55)
  audio -> Whisper-small encoder (last layer) and WavLM-large (last layer)
        -> SVR per embedding -> average                                   (audio branch, weight 0.45)
  blend = 0.55 * text + 0.45 * audio
  -> linear "stretch" learned on out-of-fold predictions -> clip to [1, 5]

Run on Kaggle (GPU T4, Internet ON):   python solution.py
Output: /kaggle/working/submission_final.csv   (filename,label for the 216 test clips, test.csv order)

The given train/test data is never modified. The only data choice is which TRAIN rows the models fit on:
the 37 clips with file number >= 5000 (all labelled 0, loud non-speech recordings, none in test) are left out.
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA = os.environ.get("SHL_DATA", "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final")
W    = os.environ.get("SHL_WORK", "/kaggle/working") + "/"
ASR_MODEL   = "openai/whisper-small"
TEXT_MODELS = ["roberta-base", "roberta-large", "microsoft/deberta-v3-base", "microsoft/deberta-v3-large"]
TAG = ASR_MODEL.split("/")[-1]
AUDIO_WEIGHT = 0.45        # final submission (LB 0.3494). 0.40 gave 0.3499, 0.30 gave 0.3538


def load_meta():
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    df = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    df["path"] = [f"{DATA}/{s}/{f}" for s, f in zip(df.split, df.filename)]   # order: train (769) then test (216)
    return train, test, df


# ----------------------------------------------------------------------------- 1) transcripts
def stage_transcripts(df):
    out = f"{W}transcripts_{TAG}.csv"
    rows, done = [], set()
    if os.path.exists(out):                                   # beech me ruka to wahin se aage badhega
        old = pd.read_csv(out, keep_default_na=False)
        rows, done = old.to_dict("records"), set(zip(old.split, old.filename))
    jobs = [(s, p, f) for s, p, f in zip(df.split, df.path, df.filename) if (s, f) not in done]
    print("transcribe karne baaki:", len(jobs))
    if jobs:
        import torch, soundfile as sf
        from transformers import pipeline
        from tqdm.auto import tqdm
        asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=0,
                       chunk_length_s=30, batch_size=16, dtype=torch.float16)

        def load(path):
            a, sr = sf.read(path)
            if a.ndim > 1: a = a.mean(axis=1)
            return {"raw": a.astype("float32"), "sampling_rate": sr}

        for i in tqdm(range(0, len(jobs), 8)):
            batch = jobs[i:i + 8]
            res = asr([load(p) for _, p, _ in batch], generate_kwargs={"language": "en", "task": "transcribe"})
            for (s, _, f), o in zip(batch, res):
                rows.append({"split": s, "filename": f, "text": o["text"].strip()})
            pd.DataFrame(rows).to_csv(out, index=False)
        del asr; torch.cuda.empty_cache()
    tr = pd.read_csv(out, keep_default_na=False)
    m = {(s, f): t for s, f, t in zip(tr.split, tr.filename, tr.text)}   # (split, filename): naam train/test me repeat hote hain
    return [m[(s, f)] for s, f in zip(df.split, df.filename)]


# ----------------------------------------------------------------------------- 2) text embeddings
def stage_text_emb(texts):
    todo = [nm for nm in TEXT_MODELS if not os.path.exists(f"{W}emb_{nm.split('/')[-1]}__{TAG}.npy")]
    if not todo: return
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm.auto import tqdm
    for nm in todo:
        tok = AutoTokenizer.from_pretrained(nm)
        mdl = AutoModel.from_pretrained(nm).to("cuda").eval()
        if "deberta" not in nm: mdl = mdl.half()
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), 16), desc=nm):
                b = tok(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                h = mdl(**b).last_hidden_state
                mk = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                out.append(((h * mk).sum(1) / mk.sum(1)).float().cpu().numpy())
        np.save(f"{W}emb_{nm.split('/')[-1]}__{TAG}.npy", np.vstack(out))
        del mdl; torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- 3) audio embeddings
def stage_audio(df):
    f1, f2 = f"{W}audio_whisper_small_L9_L12.npy", f"{W}audio_wavlm_large_L12_L18_L24.npy"
    if os.path.exists(f1) and os.path.exists(f2): return
    import torch, soundfile as sf
    from tqdm.auto import tqdm

    def wav(p):
        a, _ = sf.read(p)
        return (a.mean(1) if a.ndim > 1 else a).astype("float32")

    if not os.path.exists(f1):                                           # Whisper-small encoder
        from transformers import WhisperFeatureExtractor, WhisperModel
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        enc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to("cuda").half().eval()
        L, out = [9, 12], []
        with torch.no_grad():
            for p in tqdm(df.path, desc="whisper-small encoder"):
                a = wav(p); parts, ws = [], []
                for i in range(0, len(a), 480000):                       # 30 sec ke tukde
                    c = a[i:i + 480000]
                    if len(c) < 16000 and parts: continue
                    x = fe(c, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()
                    hs = enc(x, output_hidden_states=True).hidden_states
                    nv = max(1, int(len(c) / 16000 * 50))                # sirf asli audio wale frames
                    parts.append(np.stack([hs[l][0, :nv].float().mean(0).cpu().numpy() for l in L])); ws.append(nv)
                out.append(np.average(parts, axis=0, weights=ws))
        np.save(f1, np.stack(out)); del enc; torch.cuda.empty_cache()

    if not os.path.exists(f2):                                           # WavLM-large (fp32, pehle jaisa)
        from transformers import WavLMModel, AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        mdl = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval()
        L, out = [12, 18, 24], []                                        # 24 = last layer (jo 0.3626 me use hui)
        with torch.no_grad():
            for p in tqdm(df.path, desc="wavlm-large"):
                inp = fe(wav(p), sampling_rate=16000, return_tensors="pt").to("cuda")
                hs = mdl(**inp, output_hidden_states=True).hidden_states
                out.append(np.stack([hs[l][0].float().mean(0).cpu().numpy() for l in L]))
        arr = np.stack(out); assert np.isfinite(arr).all(), "NaN aaya"
        np.save(f2, arr); del mdl; torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- 4) final model
def stage_final(train, test, audio_weight=None):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold

    wa = AUDIO_WEIGHT if audio_weight is None else audio_weight
    n, m, y = len(train), len(test), train.label.values
    keep = train.filename.str.extract(r"(\d+)")[0].astype(int).values < 5000      # 732 normal clips
    ki = np.where(keep)[0]

    text = [np.load(f"{W}emb_{nm.split('/')[-1]}__{TAG}.npy") for nm in TEXT_MODELS]
    audio = [np.load(f"{W}audio_whisper_small_L9_L12.npy")[:, -1],               # Whisper-small, last layer
             np.load(f"{W}audio_wavlm_large_L12_L18_L24.npy")[:, -1]]            # WavLM-large, last layer
    for X in text + audio:
        assert X.shape[0] == n + m, f"row mismatch: {X.shape}"

    svr = lambda: make_pipeline(StandardScaler(), SVR(C=1, epsilon=0.1))           # fixed hyper-parameters
    folds = list(KFold(5, shuffle=True, random_state=42).split(np.arange(n)))

    def oof(X):                                      # out-of-fold predictions, fitting on normal clips only
        o = np.zeros(n)
        for tr_i, va_i in folds:
            tr_i = tr_i[keep[tr_i]]
            o[va_i] = svr().fit(X[tr_i], y[tr_i]).predict(X[va_i])
        return o

    def predict_test(X):
        return svr().fit(X[:n][keep], y[keep]).predict(X[n:])

    o  = (1 - wa) * np.mean([oof(X[:n]) for X in text], axis=0)  + wa * np.mean([oof(X[:n]) for X in audio], axis=0)
    te = (1 - wa) * np.mean([predict_test(X) for X in text], axis=0) + wa * np.mean([predict_test(X) for X in audio], axis=0)

    # SVR predictions are shrunk towards the mean: stretch them with a linear map fitted on out-of-fold predictions
    slope = np.clip(np.polyfit(o[ki], y[ki], 1)[0], 1.0, 1.5)
    intercept = y[ki].mean() - slope * o[ki].mean()
    final = np.clip(intercept + slope * te, 1, 5)

    rmse = lambda p: float(np.sqrt(((y[ki] - p[ki]) ** 2).mean()))
    print(f"OOF RMSE on normal clips: raw {rmse(np.clip(o, 1, 5)):.3f} | stretched {rmse(np.clip(intercept + slope * o, 1, 5)):.3f} "
          f"(slope {slope:.2f}; the stretched number is in-sample for these 2 parameters)")
    sub = pd.DataFrame({"filename": test.filename.values, "label": final})
    assert len(sub) == m and sub.label.notna().all()
    out = f"{W}submission_final.csv"
    sub.to_csv(out, index=False)
    print("saved", out, "| mean/std", round(final.mean(), 3), round(final.std(), 3))
    return sub


def main():
    import torch
    print("GPU:", torch.cuda.is_available())
    train, test, df = load_meta()
    texts = stage_transcripts(df)
    stage_text_emb(texts)
    stage_audio(df)
    stage_final(train, test)


if __name__ == "__main__":
    main()
