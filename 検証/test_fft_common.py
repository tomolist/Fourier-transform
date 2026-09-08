# -*- coding: utf-8 -*-
"""fft_common.py の回帰テスト

波形の立ち上がるタイミングと形状はファイルごとに変わるため、
実データを加工した異常ケースでイベント検出が壊れないことを確認する。

実行方法（プロジェクトルートから）:
    .venv/Scripts/python.exe 検証/test_fft_common.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "コードフォルダ"))
from fft_common import (                                   # noqa: E402
    ZERO_ADJUST_ROWS, amp_spectrum, find_event, metrics, psd,
    read_any, read_nr500, rise_time, robust_sigma, zero_adjust,
    zero_adjust_check,
)

SAMPLE = os.path.join(ROOT, "加工対象ファイルサンプル")
FUTABA = os.path.join(SAMPLE, "038_210℃_080_1.0mm_futaba.csv")
NR500 = os.path.join(SAMPLE, "040_210℃_080_1.0mm.csv")

_fails = []


def check(label, ok, detail=""):
    print(f"  {'OK ' if ok else 'NG '} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _fails.append(label)


def close(a, b, tol):
    return bool(np.isfinite(a)) and abs(a - b) <= tol


# ==========================================================
def test_spectrum_normalization():
    print("\n[1] スペクトルの正規化")
    dt, n = 5e-6, 20000
    t = np.arange(n) * dt
    amp, f0, off = 1.0, 100.0, 0.6          # 分解能10Hz なので 100Hz はビン中心
    y = amp * np.sin(2 * np.pi * f0 * t) + off

    for win in ("none", "hann"):
        f, S = amp_spectrum(y, dt, detrend=False, window=win)
        i = int(np.argmin(np.abs(f - f0)))
        check(f"振幅スペクトル 窓={win}: DC={S[0]:.6f}, {f0:.0f}Hz={S[i]:.6f}",
              close(S[0], off, 1e-6) and close(S[i], amp, 1e-6))

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.5, 20000)
    f, P = psd(y, 1e-3, detrend=True, window="none")
    integ = float(np.trapezoid(P, f))
    check("PSDのパーセバル則（矩形窓で分散と一致）",
          close(integ, float(y.var()), 1e-6 * y.var()),
          f"積分={integ:.6f} vs 分散={y.var():.6f}")


# ==========================================================
def test_real_files():
    print("\n[2] 実ファイルの指標（期待値は実測に基づく）")
    expect = {
        ("futaba", "CH03"): dict(peak=33.460, rise=106.0, f999=17.0, noise=0.074),
        ("futaba", "CH04"): dict(peak=22.780, rise=77.0, f999=18.1, noise=0.046),
        ("nr500", "V03"): dict(peak=27.756, rise=92.0, f999=17.3, noise=0.059),
        ("nr500", "V04"): dict(peak=22.163, rise=80.0, f999=19.2, noise=0.059),
    }
    for path in (FUTABA, NR500):
        df, dt, info = read_any(path)
        for ch in df.columns:
            e = expect[(info["format"], ch)]
            m, _, _ = metrics(df[ch].to_numpy(), dt)
            tag = f"{info['format']}/{ch}"
            check(f"{tag} ピーク={m['peak_MPa']:.3f}", close(m["peak_MPa"], e["peak"], 0.01))
            check(f"{tag} 立上り={m['rise_time_ms']:.1f}ms",
                  close(m["rise_time_ms"], e["rise"], 3.0))
            check(f"{tag} f99.9={m['f999_Hz']:.2f}Hz", close(m["f999_Hz"], e["f999"], 1.5))
            check(f"{tag} ノイズRMS={m['noise_rms_MPa']:.4f}",
                  close(m["noise_rms_MPa"], e["noise"], 0.01))
            check(f"{tag} フラグ={m['flags']}", m["flags"] == "ok")


# ==========================================================
def test_zero_adjust():
    print("\n[3] ゼロ点合わせ（先頭1000行の平均を引く・ユーザー指定仕様）")
    raw, dt = read_nr500(NR500)                      # ゼロ点合わせ前
    adj, offset = zero_adjust(raw, ZERO_ADJUST_ROWS)

    for ch, want in (("V03", -0.1756), ("V04", -1.1230)):
        check(f"{ch} 引いた値={offset[ch]:+.4f}", close(offset[ch], want, 1e-3))
        head_mean = float(adj[ch].iloc[:ZERO_ADJUST_ROWS].mean())
        check(f"{ch} 適用後の先頭1000行平均={head_mean:.2e}（厳密に0）",
              abs(head_mean) < 1e-12)
        # 波形の形（ピーク-ベースライン）が保存されること
        d_raw = float(raw[ch].max() - raw[ch].iloc[:ZERO_ADJUST_ROWS].mean())
        d_adj = float(adj[ch].max())
        check(f"{ch} ピーク-ベースラインが保存 {d_adj:.3f} MPa", close(d_adj, d_raw, 1e-9))

    check("zero_adjust は元のDataFrameを破壊しない",
          abs(float(raw["V03"].iloc[:1000].mean())) > 1e-6)

    print("\n[4] ゼロ点合わせ窓の汚染検知")
    y = raw["V03"].to_numpy() * 1.0
    r = zero_adjust_check(y, ZERO_ADJUST_ROWS)
    check(f"正常時はフラグなし（差={abs(r['zero_offset'] - r['zero_offset_robust']):.4f}）",
          r["flags"] == [])
    shifted = np.roll(y, -421000)                    # イベントを記録先頭へ
    r2 = zero_adjust_check(shifted, ZERO_ADJUST_ROWS)
    d = abs(r2["zero_offset"] - r2["zero_offset_robust"])
    check(f"イベントが先頭に重なると検知（{d / r2['noise_sigma']:.0f}シグマ）",
          "zero_window_offset" in r2["flags"])


# ==========================================================
def test_waveform_variation():
    print("\n[5] 波形のばらつきへの耐性（実データを加工した8ケース）")
    fut = pd.read_csv(FUTABA, header=2).drop(0).reset_index(drop=True)
    y0 = pd.to_numeric(fut["CH03"]).to_numpy(float)
    dt = 0.001
    quiet = y0[:1000].mean()

    spike = y0.copy()
    spike[500] = y0.max() * 0.35                     # イベント前に単発スパイク

    two = np.concatenate([y0[2000:4000], np.full(500, quiet), y0[2000:4000]])
    cases = [
        ("1. 正常", y0, None, 106.0),
        ("2. イベントが記録の先頭", np.roll(y0, -2500), "event_at_edge", 106.0),
        ("3. イベントが記録の末尾", np.roll(y0, 16800), None, 106.0),
        ("4. イベントなし", y0[:2000].copy(), "no_event", None),
        ("5. 2イベント", two, "multiple_events", 107.0),
        ("6. 弱いショット(1/20)", (y0 - quiet) / 20 + quiet, None, 106.0),
        ("7. 事前にノイズスパイク", spike, "multiple_events", 106.0),
        ("8. ベースラインドリフト", y0 + np.linspace(0, 0.5, len(y0)), None, 105.0),
    ]
    for name, y, want_flag, want_rise in cases:
        i0, i1, flags, info = find_event(y, dt)
        fl = ",".join(flags) if flags else "ok"
        if want_flag is None:
            ok_flag = all("low_snr" in f for f in flags)
        else:
            ok_flag = any(want_flag in f for f in flags)
        detail = f"区間={i0 * dt:.3f}-{i1 * dt:.3f}s フラグ={fl}"
        if want_rise is None:
            check(name, ok_flag, detail)
        else:
            tr = rise_time(y, dt, i0, i1) * 1000
            check(f"{name} 立上り={tr:.1f}ms", ok_flag and close(tr, want_rise, 3.0), detail)

    # 最重要：素朴な実装（先頭から最初の交差）だと 2000ms超になるケース
    i0, i1, _, _ = find_event(spike, dt)
    tr = rise_time(spike, dt, i0, i1) * 1000
    ip = int(np.argmax(spike))
    pre = spike[:ip + 1]
    a = int(np.flatnonzero(pre >= 0.1 * spike.max())[0])
    b = int(np.flatnonzero(pre >= 0.9 * spike.max())[0])
    naive = (b - a) * dt * 1000
    check(f"スパイク混入時: 堅牢={tr:.1f}ms / 素朴={naive:.1f}ms",
          close(tr, 106.0, 3.0) and naive > 1000,
          "素朴な実装は20倍の誤差になる")


# ==========================================================
def test_robust_sigma():
    print()
    print("[6] 量子化データでのノイズ推定（MADが0になる場合のフォールバック）")
    # Futaba は 0.01MPa 刻み。静穏部では過半が同じ値になり MAD が 0 になる。
    quantized = np.round(np.random.default_rng(1).normal(0, 0.004, 5000), 2)
    mad = float(np.median(np.abs(quantized - np.median(quantized))))
    sig = robust_sigma(quantized)
    check(f"MAD={mad:.4f} でも sigma={sig:.4f} と 0 にならない", sig > 0,
          "フォールバックがないとノイズ低減が 0 倍と誤判定される")

    # 実ファイルの静穏部でも 0 にならないこと
    fut = pd.read_csv(FUTABA, header=2).drop(0).reset_index(drop=True)
    for ch in ("CH03", "CH04"):
        seg = pd.to_numeric(fut[ch]).to_numpy(float)[:2000]
        check(f"{ch} 静穏部 sigma={robust_sigma(seg):.4f} > 0", robust_sigma(seg) > 0)


# ==========================================================
def test_no_duplicate_columns():
    print("\n[7] metrics に重複列がないこと（CSVの列が二重にならない）")
    for path in (FUTABA, NR500):
        df, dt, info = read_any(path)
        ch = df.columns[0]
        m, _, _ = metrics(df[ch].to_numpy(), dt,
                          zero_rows=ZERO_ADJUST_ROWS if info["zero_applied"] else None)
        keys = list(m)
        check(f"{info['format']}: 列名の重複なし（{len(keys)}列）",
              len(keys) == len(set(keys)))

        # 名前が違っても中身が常に同じ列がないかを値で確認する
        same = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                va, vb = m[a], m[b]
                if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
                        and not isinstance(va, bool) and not isinstance(vb, bool)
                        and np.isfinite(va) and np.isfinite(vb)
                        and va == vb and va != 0):
                    same.append(f"{a}=={b}")
        check(f"{info['format']}: 同じ値を持つ列の組がない",
              not same, f"({', '.join(same)})" if same else "")


# ==========================================================
def test_readonly():
    print("\n[8] 加工対象ファイルを書き換えないこと")
    before = {}
    for fn in sorted(os.listdir(SAMPLE)):
        with open(os.path.join(SAMPLE, fn), "rb") as f:
            before[fn] = hashlib.sha256(f.read()).hexdigest()

    for path in (FUTABA, NR500):                     # 一通り読み込んで解析する
        df, dt, info = read_any(path)
        for ch in df.columns:
            metrics(df[ch].to_numpy(), dt, zero_rows=ZERO_ADJUST_ROWS)

    for fn, want in before.items():
        with open(os.path.join(SAMPLE, fn), "rb") as f:
            now = hashlib.sha256(f.read()).hexdigest()
        check(f"{fn} のSHA-256が不変", now == want)


# ==========================================================
if __name__ == "__main__":
    test_spectrum_normalization()
    test_real_files()
    test_zero_adjust()
    test_waveform_variation()
    test_robust_sigma()
    test_no_duplicate_columns()
    test_readonly()
    print("\n" + "=" * 60)
    if _fails:
        print(f"NG {len(_fails)} 件:")
        for name in _fails:
            print("  -", name)
        sys.exit(1)
    print("全テスト OK")
