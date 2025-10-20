import os
import argparse
import random
import string
import numpy as np
import tensorflow as tf
import soundfile as sf

#!/usr/bin/env python3
# /home/eavi/code/projects/PokeShout/train_ai.py
# GitHub Copilot


# Config
SAMPLE_RATE = 16000
DURATION = 1.2  # seconds for output audio
AUDIO_LEN = int(SAMPLE_RATE * DURATION)
MAX_NAME_LEN = 13
VOCAB = " " + string.ascii_lowercase + "-'"
VOCAB_MAP = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
EMBED_DIM = 32
BATCH_SIZE = 32
DEFAULT_TRAIN_SAMPLES = 2000
MODEL_PATH = "name2audio_model.h5"


def name_to_intseq(name, max_len=MAX_NAME_LEN):
    name = name.lower()
    seq = [VOCAB_MAP.get(c, 0) for c in name[:max_len]]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return np.array(seq, dtype=np.int32)


def synth_target_audio(name, sr=SAMPLE_RATE, out_len=AUDIO_LEN):
    # simple rule-based "spoken-like" synthetic audio for training targets
    name = name.lower()
    if len(name) == 0:
        return np.zeros(out_len, dtype=np.float32)
    num_chars = max(1, len(name))
    seg_len = out_len // num_chars
    audio = np.zeros(out_len, dtype=np.float32)
    t_seg = np.linspace(0, seg_len / sr, seg_len, endpoint=False)
    base_freq = 120.0
    for i, ch in enumerate(name):
        # map char to a frequency
        if ch in string.ascii_lowercase:
            idx = ord(ch) - ord('a')
            freq = base_freq + idx * 12 + (i % 3) * 15
        elif ch == ' ':
            freq = base_freq * 0.5
        else:
            freq = base_freq + (i % 5) * 30
        # vowel-like richer tone
        seg = 0.6 * np.sin(2 * np.pi * freq * t_seg)
        seg += 0.25 * np.sin(2 * np.pi * (freq * 1.9) * t_seg)
        seg += 0.1 * np.sin(2 * np.pi * (freq * 2.7) * t_seg)
        # amplitude envelope
        env = np.sin(np.pi * np.linspace(0, 1, seg_len)) ** 0.8
        seg *= env
        start = i * seg_len
        end = start + seg_len
        if end > out_len:
            end = out_len
            seg = seg[: end - start]
        audio[start:end] = seg
    # small global smoothing and normalization
    audio = audio * 0.95 / (np.max(np.abs(audio)) + 1e-9)
    return audio.astype(np.float32)


def generate_dataset(n_samples=DEFAULT_TRAIN_SAMPLES):
    """
    Backwards-compatible: generate synthetic name/audio pairs (as before).
    """
    names = []
    xs = []
    ys = []
    for _ in range(n_samples):
        # random name-like string
        length = random.randint(2, 10)
        name = ""
        for i in range(length):
            if random.random() < 0.15:
                name += " "
            else:
                name += random.choice(string.ascii_lowercase)
        name = name.strip()
        if name == "":
            name = "a"
        seq = name_to_intseq(name)
        audio = synth_target_audio(name)
        names.append(name)
        xs.append(seq)
        ys.append(audio)
    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys


def generate_dataset_from_dir(data_dir, n_samples=None):
    """
    Load audio files from data_dir. Use file basename (without extension) as the name input.
    Returns names, xs, ys stacked arrays. If n_samples is provided, take up to that many files.
    Supported extensions: wav, ogg, flac, mp3.
    Files are converted to mono and resampled/padded/truncated to AUDIO_LEN samples.
    """
    exts = (".wav", ".ogg", ".flac", ".mp3")
    files = [f for f in sorted(os.listdir(data_dir))
             if f.lower().endswith(exts)]
    if n_samples:
        files = files[:n_samples]

    names = []
    xs = []
    ys = []

    for fn in files:
        path = os.path.join(data_dir, fn)
        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception as e:
            print(f"Skipping {fn}: read error: {e}")
            continue
        # mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # resample/interpolate to target AUDIO_LEN
        if len(data) == 0:
            continue
        # create resampled array of length AUDIO_LEN
        orig_len = len(data)
        target_len = AUDIO_LEN
        if orig_len == target_len:
            out = data
        else:
            # linear interpolation over sample indices
            xp = np.linspace(0, orig_len, orig_len, endpoint=False)
            x = np.linspace(0, orig_len, target_len, endpoint=False)
            out = np.interp(x, xp, data).astype(np.float32)

        # ensure final length exactly AUDIO_LEN
        if out.shape[0] < AUDIO_LEN:
            pad = np.zeros(AUDIO_LEN - out.shape[0], dtype=np.float32)
            out = np.concatenate([out, pad])
        elif out.shape[0] > AUDIO_LEN:
            out = out[:AUDIO_LEN]

        name = os.path.splitext(fn)[0]
        name = name.replace("_", " ").strip().lower()[:MAX_NAME_LEN]
        if name == "":
            name = "a"
        seq = name_to_intseq(name)
        names.append(name)
        xs.append(seq)
        ys.append(out)

    if not xs:
        return [], np.zeros((0, MAX_NAME_LEN), dtype=np.int32), np.zeros((0, AUDIO_LEN), dtype=np.float32)

    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys


def build_model(max_name_len=MAX_NAME_LEN, vocab_size=VOCAB_SIZE, audio_len=AUDIO_LEN):
    inp = tf.keras.Input(shape=(max_name_len,), dtype=tf.int32, name="name_in")
    x = tf.keras.layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96))(x)
    x = tf.keras.layers.Dense(768, activation="relu")(x)
    x = tf.keras.layers.Dense(audio_len, activation="tanh")(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="name2audio")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_and_save(model_path=MODEL_PATH, epochs=30, samples=DEFAULT_TRAIN_SAMPLES, data_dir=None):
    """
    If data_dir is provided and contains audio files, use them as dataset.
    Otherwise fallback to synthetic generate_dataset.
    """
    if data_dir:
        print(f"Loading dataset from directory: {data_dir}")
        _, xs, ys = generate_dataset_from_dir(data_dir, n_samples=samples)
    else:
        _, xs, ys = generate_dataset(samples)

    if xs.shape[0] == 0:
        raise RuntimeError(
            "No training samples available (check data_dir or increase samples).")

    model = build_model()
    model.fit(xs, ys, batch_size=BATCH_SIZE,
              epochs=epochs, validation_split=0.05)
    model.save(model_path)
    return model


def denoise_audio(audio, sr=SAMPLE_RATE, cutoff_hz=4000):
    """
    Very simple FFT low-pass denoiser: zeroes frequency bins above cutoff_hz.
    - audio: 1D numpy array (float32)
    - sr: sample rate
    - cutoff_hz: cutoff frequency in Hz
    Returns a float32 array same length as input.
    """
    if cutoff_hz <= 0 or cutoff_hz >= sr // 2:
        return audio
    n = len(audio)
    # real-FFT, zero bins above cutoff
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    A = np.fft.rfft(audio)
    A[freqs > cutoff_hz] = 0
    out = np.fft.irfft(A, n)
    return out.astype(np.float32)


def synthesize(name, model, out_path, denoise=False, denoise_cutoff=4000):
    # build fixed-length sequence matching the model's expected input length
    seq = name_to_intseq(name)
    expected_len = model.input_shape[1] or MAX_NAME_LEN
    if seq.shape[0] < expected_len:
        pad = np.zeros(expected_len - seq.shape[0], dtype=np.int32)
        seq = np.concatenate([seq, pad])
    elif seq.shape[0] > expected_len:
        seq = seq[:expected_len]

    seq = np.expand_dims(seq, axis=0)
    pred = model.predict(seq)[0].astype(np.float32)

    # normalize to [-0.95, 0.95]
    maxv = np.max(np.abs(pred)) + 1e-9
    pred = pred / maxv * 0.95

    if denoise:
        pred = denoise_audio(pred, SAMPLE_RATE, denoise_cutoff)
        # re-normalize after denoising
        maxv = np.max(np.abs(pred)) + 1e-9
        pred = pred / maxv * 0.95

    sf.write(out_path, pred, SAMPLE_RATE, format="OGG", subtype="VORBIS")
    print(f"Wrote {out_path} (denoise={denoise}, cutoff={denoise_cutoff} Hz)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a toy TF model that maps a name to an OGG audio file.")
    parser.add_argument("--train", action="store_true",
                        help="Train a model before synthesizing.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs if --train is set.")
    parser.add_argument("--samples", type=int, default=DEFAULT_TRAIN_SAMPLES,
                        help="Number of synthetic training samples.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with audio files (basename used as name input).")
    parser.add_argument("--name", type=str, default="alice",
                        help="Name to synthesize.")
    parser.add_argument("--out", type=str, default="out.ogg",
                        help="Output OGG file path.")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to save/load the model.")
    parser.add_argument("--denoise", action="store_true",
                        help="Apply low-pass denoising on synthesized output")
    parser.add_argument("--denoise-cutoff", type=int, default=4000,
                        help="Denoise low-pass cutoff frequency (Hz)")
    args = parser.parse_args()

    if args.train or not os.path.exists(args.model):
        print("Training model...")
        model = train_and_save(model_path=args.model, epochs=args.epochs,
                               samples=args.samples, data_dir=args.data_dir)
    else:
        model = tf.keras.models.load_model(args.model)

    print(f"Synthesizing name '{args.name}' -> {args.out}")
    synthesize(args.name, model, args.out, denoise=args.denoise,
               denoise_cutoff=args.denoise_cutoff)
    print("Done.")


if __name__ == "__main__":
    main()
