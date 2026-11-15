#!/usr/bin/env python3
"""
Sonify a global GeoTIFF (GEDI RH98 canopy height) as a WAV file.

  1. Split the map into an 8 (rows) x 120 (cols) grid; each cell = mean of valid pixels.
  2. Pad (continuous synth): each of the 8 rows is one voice with a fixed pitch.
     The cell mean controls loudness + brightness. No-data cells fade to silence.
  3. Melody (piano): the median of each column (over cells that have data),
     mapped to a D-minor-pentatonic scale. One note per column, left (west) to right (east).
     Columns with no data at all are rests.

Usage:
  pip install numpy scipy rasterio
  python sonify_gedi.py --tif gedi_rh98_1000m.tif --out gedi_sonification.wav
"""
import argparse, os
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, butter, sosfilt

SR = 44100
NR, NC = 8, 120


# ----------------------------------------------------------------- grid
def compute_grid(tif, nr=NR, nc=NC, step=256):
    """Mean of valid pixels per cell (NaN where a cell has no valid pixel)."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(tif) as src:
        H, W, nod = src.height, src.width, src.nodata
        rows = np.linspace(0, H, nr + 1).round().astype(int)
        cols = np.linspace(0, W, nc + 1).round().astype(int)
        S = np.zeros((nr, nc)); C = np.zeros((nr, nc))
        for r in range(nr):
            for y in range(rows[r], rows[r + 1], step):
                h = min(step, rows[r + 1] - y)
                a = src.read(1, window=Window(0, y, W, h)).astype("float64")
                valid = np.isfinite(a)
                if nod is not None:
                    valid &= a != nod
                a = np.where(valid, a, 0.0)
                S[r] += np.add.reduceat(a.sum(axis=0), cols[:-1])
                C[r] += np.add.reduceat(valid.sum(axis=0).astype("float64"), cols[:-1])
    return np.where(C > 0, S / np.maximum(C, 1), np.nan)


# ----------------------------------------------------------------- helpers
def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def reverb_ir(rt60=2.6, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t * 6.9 / rt60)
    ir = sosfilt(butter(2, 5000, btype="low", fs=sr, output="sos"), ir)
    ir[: int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))  # soft onset
    return ir / np.sqrt(np.sum(ir ** 2))


def add_reverb(stereo, wet, sr=SR):
    out = np.empty_like(stereo)
    for ch in range(2):
        w = fftconvolve(stereo[:, ch], reverb_ir(sr=sr, seed=ch + 1))[: len(stereo)]
        out[:, ch] = (1 - wet) * stereo[:, ch] + wet * w * 0.6
    return out


def smooth_envelope(t, centres, values, sigma_sec, ctrl_rate=200):
    """Interpolate column values over time, smooth at a low control rate, upsample to audio rate."""
    tc = np.arange(0, t[-1] + 1 / ctrl_rate, 1 / ctrl_rate)
    e = gaussian_filter1d(np.interp(tc, centres, values), sigma=sigma_sec * ctrl_rate)
    return np.interp(t, tc, e)


def pan_gains(p):  # p in [-1, 1], equal power
    a = (p + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


# ----------------------------------------------------------------- pad
# MIDI pitch per row (row 0 = north = highest). Rows 1-6 hold the data for GEDI.
PAD_MIDI = [67, 64, 60, 57, 53, 50, 45, 38]  # G4 E4 C4 A3 F3 D3 A2 D2


def render_pad(grid, col_sec, n_samples, vmin, vmax, n_harm=10):
    nr, nc = grid.shape
    t = np.arange(n_samples) / SR
    centres = (np.arange(nc) + 0.5) * col_sec
    norm = (grid - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    amp_cols = np.where(np.isnan(grid), 0.0, 0.25 + 0.75 * norm)   # no data -> silent
    bri_cols = np.where(np.isnan(grid), 0.3, 0.35 + 0.5 * norm)    # taller canopy -> brighter
    pans = np.linspace(-0.6, 0.6, nr)
    mix = np.zeros((n_samples, 2))
    for r in range(nr):
        if not np.any(amp_cols[r] > 0):
            continue
        env = smooth_envelope(t, centres, amp_cols[r], 0.12)        # smooth fades
        bri = smooth_envelope(t, centres, bri_cols[r], 0.2)
        voice = np.zeros(n_samples)
        for detune in (-6, +6):                                    # 2 slightly detuned oscillators
            f = mtof(PAD_MIDI[r] + detune / 100)
            phi = 2 * np.pi * f * t
            for h in range(1, n_harm + 1):
                if f * h > 9000:
                    break
                voice += (bri ** (h - 1) / h) * np.sin(h * phi)
        voice *= env
        gl, gr = pan_gains(pans[r])
        mix[:, 0] += gl * voice
        mix[:, 1] += gr * voice
    return mix


# ----------------------------------------------------------------- piano
def piano_note(midi, hold, release=0.35, vel=0.8, rng=None):
    f0 = float(mtof(midi))
    dur = hold + release
    n = int(dur * SR)
    t = np.arange(n) / SR
    B = 1.5e-4 * 2 ** ((midi - 60) / 18)                           # inharmonicity
    T0 = 5.0 * 2 ** (-(midi - 60) / 20)                            # higher notes die faster
    out = np.zeros(n)
    k = 1
    while k <= 28:
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 11000:
            break
        a = (1 / k ** 1.15) * (abs(np.sin(np.pi * k * 0.125)) + 0.2)  # hammer strike position
        a *= vel ** (1 + 0.04 * k)                                    # soft notes = duller
        tau = T0 / (1 + 0.7 * (k - 1))
        env = 0.6 * np.exp(-t / (0.3 * tau)) + 0.4 * np.exp(-t / tau)  # fast + slow decay stages
        beat = 0.25 + 0.05 * k                                         # 2 strings, slight beating
        out += a * env * 0.5 * (np.sin(2 * np.pi * fk * t) + np.sin(2 * np.pi * (fk + beat) * t))
        k += 1
    # hammer thump
    rng = rng or np.random.default_rng(0)
    nn = int(0.02 * SR)
    thump = rng.standard_normal(nn) * np.exp(-np.arange(nn) / (0.004 * SR))
    thump = sosfilt(butter(2, [200, 3000], btype="band", fs=SR, output="sos"), thump)
    out[:nn] += 0.08 * vel * thump
    # attack + damper release
    na = int(0.003 * SR)
    out[:na] *= np.linspace(0, 1, na)
    nr_ = int(release * SR)
    out[-nr_:] *= np.exp(-np.linspace(0, 6, nr_))
    return out


def render_melody(medians, col_sec, n_samples, vmin, vmax, hold_cols=1.5):
    # D minor pentatonic, D4 (62) up to D6 (86)
    scale = [m for m in range(62, 87) if (m - 62) % 12 in (0, 3, 5, 7, 10)]
    rng = np.random.default_rng(42)
    mono = np.zeros(n_samples)
    notes = []
    for j, v in enumerate(medians):
        if np.isnan(v):
            notes.append(None)
            continue
        idx = int(round((v - vmin) / (vmax - vmin) * (len(scale) - 1)))
        midi = scale[int(np.clip(idx, 0, len(scale) - 1))]
        notes.append(midi)
        nt = piano_note(midi, hold=hold_cols * col_sec, rng=rng)
        s = int(j * col_sec * SR)
        e = min(n_samples, s + len(nt))
        mono[s:e] += nt[: e - s]
    gl, gr = pan_gains(0.0)
    return np.stack([mono * gl, mono * gr], axis=1), notes


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", default="gedi_rh98_1000m.tif")
    ap.add_argument("--out", default="gedi_sonification.wav")
    ap.add_argument("--grid-csv", default="gedi_grid_8x120.csv",
                    help="cache of the 8x120 mean grid (computed from the tif if missing)")
    ap.add_argument("--col-seconds", type=float, default=0.5, help="duration of one grid column")
    args = ap.parse_args()

    if os.path.exists(args.grid_csv):
        grid = np.genfromtxt(args.grid_csv, delimiter=",")
    else:
        grid = compute_grid(args.tif)
        np.savetxt(args.grid_csv, grid, delimiter=",", fmt="%.4f")
    print("grid", grid.shape, "valid cells:", int((~np.isnan(grid)).sum()))

    with np.errstate(all="ignore"):
        medians = np.array([np.nanmedian(grid[:, j]) if np.any(~np.isnan(grid[:, j])) else np.nan
                            for j in range(grid.shape[1])])
    gmin, gmax = np.nanmin(grid), np.nanmax(grid)
    mmin, mmax = np.nanmin(medians), np.nanmax(medians)

    tail = 3.0
    total = grid.shape[1] * args.col_seconds + tail
    n = int(total * SR)

    pad = add_reverb(render_pad(grid, args.col_seconds, n, gmin, gmax), wet=0.35)
    piano, notes = render_melody(medians, args.col_seconds, n, mmin, mmax)
    piano = add_reverb(piano, wet=0.25)

    # balance: pad sits underneath the piano
    rms = lambda x: np.sqrt(np.mean(x[np.abs(x) > 1e-6] ** 2))
    pad *= 0.6 * rms(piano) / rms(pad)
    mix = pad + piano
    fade = int(0.05 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)[:, None]
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix *= 0.89 / np.max(np.abs(mix))                                # -1 dBFS peak
    wavfile.write(args.out, SR, (mix * 32767).astype(np.int16))
    print(f"wrote {args.out}  ({total:.1f} s, stereo, {SR} Hz, 16-bit)")
    print("notes played:", sum(x is not None for x in notes), "| rests:", sum(x is None for x in notes))


if __name__ == "__main__":
    main()
