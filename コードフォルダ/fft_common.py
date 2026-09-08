# -*- coding: utf-8 -*-
"""圧力経時変化データのフーリエ変換 共通モジュール

Futaba形式 と KEYENCE NR-500形式 の2種類のCSVを読み、
ローパスフィルタのカットオフ周波数を決めるための解析を行う。
ノートブック 01〜07 から共通で利用する。

★重要★ 加工対象CSVには一切書き込まない（読み取り専用）。
ゼロ点合わせ・DC除去・窓関数はすべてメモリ上の配列に対する操作であり、
元ファイルには書き戻さない。
"""

import os
import re

import numpy as np
import pandas as pd

# numpy 2.0 で trapz が trapezoid に改名されたため両対応
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ==========================================================
# 定数
# ==========================================================
VOLT_TO_MPA = 20.0       # ★ 圧力[MPa] = 電圧[V] × 20（NR-500のみ。スケーリングOFFのため）
ZERO_ADJUST_ROWS = 1000  # ★ ゼロ点合わせに使う先頭行数（ユーザー指定仕様）

FUTABA_CHANNELS = ("CH03", "CH04")
NR500_CHANNELS = ("V03", "V04")   # 実際の列名 "(1)HA-V03" / "(1)HA-V04" に末尾一致で解決

# チャンネルの対応（Futaba ⇔ NR-500）
CHANNEL_ALIAS = {"CH03": "V03", "CH04": "V04", "V03": "CH03", "V04": "CH04"}


# ==========================================================
# 形式判定と読み込み
# ==========================================================
def detect_format(path):
    """1行目を見て "futaba" / "nr500" / None を返す（Noneは対象外ファイル）。"""
    try:
        with open(path, "r", encoding="cp932", errors="replace") as f:
            first = f.readline()
    except OSError:
        return None
    if first.startswith("#BeginHeader"):
        return "nr500"
    if first.startswith("Time:"):
        return "futaba"
    return None


def probe_nr500(path):
    """NR-500のヘッダだけを読み、skiprows / dt / データ数 を返す。違う形式なら None。"""
    with open(path, "r", encoding="cp932", errors="replace") as f:
        head = [f.readline().rstrip("\n") for _ in range(120)]

    if not head[0].startswith("#BeginHeader"):
        return None

    # "#BeginHeader,71" → ヘッダ71行。列名行はその71行目なので skiprows=70
    n_header = int(head[0].split(",")[1])

    # "サンプリング周期,5μs" を秒に変換
    # （16行目の "実サンプリング周期" は先頭が「実」なので startswith で誤マッチしない）
    raw = next(l.split(",")[1] for l in head if l.startswith("サンプリング周期"))
    m = re.match(r"\s*([\d.]+)\s*(μs|us|ms|s)\s*$", raw)
    if m is None:
        raise ValueError(f"サンプリング周期を解釈できません: {raw!r}")
    dt = float(m.group(1)) * {"μs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}[m.group(2)]

    # "データ数,1000000" → これを nrows に使い、末尾3行のフッタ
    # （#BeginMark,3 / CH名,... / #EndMark）を構造的に除外する
    n_data = int(next(l.split(",")[1] for l in head if l.startswith("データ数")))

    return {"skiprows": n_header - 1, "dt": dt, "n_data": n_data}


def read_futaba(path, channels=FUTABA_CHANNELS):
    """Futaba形式CSVを読む。単位は元から MPa なので換算しない。"""
    df = pd.read_csv(path, header=2)      # 3行目が列名
    df = df.drop(0).reset_index(drop=True)  # 4行目は単位(Unit)行なので削除

    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"列 {missing} がありません（実際の列: {list(df.columns)}）")

    out = df[list(channels)].apply(pd.to_numeric, errors="coerce").dropna()
    return out.astype("float64").reset_index(drop=True), 0.001


def read_nr500(path, channels=NR500_CHANNELS):
    """NR-500形式CSVを読み、指定チャンネルを MPa に換算して返す。"""
    meta = probe_nr500(path)
    if meta is None:
        raise ValueError("NR-500形式ではありません")

    header = pd.read_csv(
        path, skiprows=meta["skiprows"], encoding="cp932", nrows=0
    ).columns.tolist()

    # "V03" → "(1)HA-V03" を末尾一致で解決（ユニット番号が変わっても追従できる）
    resolved = {}
    for ch in channels:
        hit = [c for c in header if c.endswith(ch)]
        if not hit:
            raise ValueError(f"チャンネル '{ch}' が見つかりません（実際の列: {header}）")
        resolved[ch] = hit[0]

    read_kw = dict(
        skiprows=meta["skiprows"],
        encoding="cp932",
        usecols=list(resolved.values()),
        nrows=meta["n_data"],   # フッタ3行を構造的に除外
    )
    try:
        # dtype を明示しないと pandas 3.0.3 では usecols 使用時に
        # IndexError: list index out of range が出るため、必ず指定する
        df = pd.read_csv(path, dtype="float32", **read_kw)
    except ValueError:
        # 数値以外が混入していた場合の保険
        df = pd.read_csv(path, dtype=str, **read_kw)
        df = df.apply(pd.to_numeric, errors="coerce").dropna().astype("float32")

    df = df.rename(columns={v: k for k, v in resolved.items()})[list(resolved)]
    return (df.astype("float64") * VOLT_TO_MPA).reset_index(drop=True), meta["dt"]


def read_any(path, channels=None, zero=None, zero_rows=ZERO_ADJUST_ROWS):
    """形式を自動判別して読み、MPa単位の DataFrame と dt を返す。

    戻り値: (df, dt, info)
      info["format"]       … "futaba" / "nr500"
      info["zero_applied"] … ゼロ点合わせを実施したか
      info["zero_offset"]  … チャンネルごとに引いた値（dict, MPa）

    zero: None のとき NR-500 は True、Futaba は False（既定の使い分け）
    """
    fmt = detect_format(path)
    if fmt == "nr500":
        df, dt = read_nr500(path, channels or NR500_CHANNELS)
        if zero is None:
            zero = True
    elif fmt == "futaba":
        df, dt = read_futaba(path, channels or FUTABA_CHANNELS)
        if zero is None:
            zero = False
    else:
        raise ValueError("Futaba形式でもNR-500形式でもありません")

    info = {"format": fmt, "zero_applied": False, "zero_offset": {}}
    if zero:
        df, offset = zero_adjust(df, zero_rows)
        info["zero_applied"] = True
        info["zero_offset"] = offset
    return df, dt, info


# ==========================================================
# ゼロ点合わせ（ユーザー指定仕様：先頭 n 行の平均を全行から引く）
# ==========================================================
def zero_adjust(df, n=ZERO_ADJUST_ROWS):
    """最初の n 行の平均を全行から引く。

    元ファイルには触れず、新しい DataFrame を返す。
    戻り値: (ゼロ点合わせ後のDataFrame, 引いた値のdict)
    """
    n = min(int(n), len(df))
    offset = df.iloc[:n].mean()
    return df - offset, {c: float(offset[c]) for c in df.columns}


def zero_adjust_check(y, n=ZERO_ADJUST_ROWS):
    """ゼロ点合わせに使う窓がイベントに汚染されていないかを検査する。

    指定どおり「先頭 n 行の平均」は常にそれで計算する（仕様は変えない）。
    同時に堅牢推定（中央値）も求め、両者が大きく食い違えばフラグを立てる。
    """
    n = min(int(n), len(y))
    spec = float(np.mean(y[:n]))
    rob, sigma = robust_baseline(y)
    win_std = float(np.std(y[:n]))

    flags = []
    if sigma > 0 and abs(spec - rob) > 5 * sigma:
        flags.append("zero_window_offset")
    if sigma > 0 and win_std > 5 * sigma:
        flags.append("zero_window_noisy")
    return {
        "zero_offset": spec,
        "zero_offset_robust": float(rob),
        "zero_window_std": win_std,
        "noise_sigma": float(sigma),
        "flags": flags,
    }


# ==========================================================
# 波形のばらつきに強いイベント検出
# ==========================================================
def robust_baseline(y):
    """中央値とMADで基準線とノイズ尺度を推定する。

    イベントは記録の 2〜8% しか占めないため、イベントが記録のどこにあっても
    （先頭でも末尾でも）正しい基準線が得られる。
    「記録の前半は静穏」という仮定を置かないのが要点。
    """
    y = np.asarray(y, dtype=np.float64)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(y))
    return med, sigma


def find_event(y, dt, k_noise=10.0, frac=0.05, gap_s=0.05,
               pad_ratio=0.25, pad_min_s=0.02, snr_min=20.0, min_points=64):
    """過渡パルスの区間を検出する。

    戻り値: (i0, i1, flags, info)   ※ i0, i1 は余裕(pad)を含む解析区間

    ・しきい値 = 基準線 + max(k_noise×σ, frac×(ピーク−基準線))
    ・gap_s 未満の隙間は同一イベントとして連結（波形が二山でも分断しない）
    ・候補が複数あるときは振幅ではなく「面積（エネルギー）」が最大のものを採用
      → イベント前のノイズスパイクを本体と取り違えない
    ・検出できない場合は記録全体にフォールバックし、必ずフラグで知らせる
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    flags = []
    base, sigma = robust_baseline(y)
    peak = float(y.max())
    amp = peak - base
    snr = amp / sigma if sigma > 0 else np.inf
    info = {"baseline": base, "noise_sigma": sigma, "peak": peak,
            "amp": amp, "snr": snr, "n_runs": 0}

    thr = base + max(k_noise * sigma, frac * amp)
    on = y > thr
    if not on.any():
        flags.append("no_event")
        info["event_start"] = 0.0
        info["event_end"] = (n - 1) * dt
        info["event_width"] = (n - 1) * dt
        return 0, n - 1, flags, info

    # しきい値超えの連続区間を抽出
    d = np.diff(on.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        ends.append(n - 1)

    # gap_s 未満の隙間は連結する
    gap = max(int(round(gap_s / dt)), 1)
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    info["n_runs"] = len(merged)
    if len(merged) > 1:
        flags.append(f"multiple_events({len(merged)})")

    # 面積（エネルギー）が最大の区間を本体とみなす
    areas = [_trapz(np.clip(y[s:e + 1] - base, 0, None), dx=dt) for s, e in merged]
    i0, i1 = merged[int(np.argmax(areas))]

    width = (i1 - i0) * dt
    info["event_start"] = i0 * dt
    info["event_end"] = i1 * dt
    info["event_width"] = width

    pad = int(round(max(pad_ratio * width, pad_min_s) / dt))
    a, b = i0 - pad, i1 + pad
    if a < 0 or b > n - 1:
        flags.append("event_at_edge")
    a, b = max(a, 0), min(b, n - 1)

    if snr < snr_min:
        flags.append(f"low_snr({snr:.1f})")
    if (b - a + 1) < min_points:
        flags.append("short_event")
        return 0, n - 1, flags, info      # FFTに足りないので記録全体へフォールバック

    return a, b, flags, info


def rise_time(y, dt, i0=None, i1=None, baseline=None):
    """立ち上がり時間(10-90%)を秒で返す。

    ★ピーク位置から遡って探す★ のが要点。
    記録の先頭から最初に10%を超えた点を探す素朴な実装は、イベント前に
    ノイズスパイクが1つあるだけで破綻する（実測で20倍の誤差が出た）。
    """
    y = np.asarray(y, dtype=np.float64)
    seg = y if i0 is None else y[i0:i1 + 1]
    if len(seg) < 3:
        return float("nan")
    if baseline is None:
        baseline, _ = robust_baseline(y)

    ip = int(np.argmax(seg))
    pk = seg[ip] - baseline
    if pk <= 0:
        return float("nan")
    hi, lo = baseline + 0.9 * pk, baseline + 0.1 * pk

    j = ip
    while j > 0 and seg[j] > hi:      # 最後に90%を下回った点
        j -= 1
    k = j
    while k > 0 and seg[k] > lo:      # そこから遡って最後に10%を下回った点
        k -= 1
    return (j - k) * dt


# ==========================================================
# スペクトル
# ==========================================================
def _prep(y, detrend=True, window="hann"):
    """FFT前の下ごしらえ。窓関数の補正係数も返す。"""
    y = np.asarray(y, dtype=np.float64)
    if detrend:
        y = y - y.mean()
    if window == "hann":
        w = np.hanning(len(y))
        return y * w, float(w.mean()), float((w ** 2).mean())
    return y, 1.0, 1.0


def amp_spectrum(y, dt, detrend=True, window="hann"):
    """片側の振幅スペクトル [MPa] を返す。"""
    n = len(y)
    yy, acg, _ = _prep(y, detrend, window)
    A = np.abs(np.fft.rfft(yy)) / (n / 2) / acg
    A[0] /= 2
    if n % 2 == 0 and len(A) > 1:
        A[-1] /= 2          # Nyquist成分も片側化で2倍しない
    return np.fft.rfftfreq(n, d=dt), A


def psd(y, dt, detrend=True, window="hann"):
    """片側のパワースペクトル密度 [MPa^2/Hz] を返す。

    振幅スペクトルと違い Δf に依存しないので、サンプリング周期の異なる
    Futaba形式 と NR-500形式 を直接比較できる。
    """
    n = len(y)
    yy, _, pcg = _prep(y, detrend, window)
    P = (np.abs(np.fft.rfft(yy)) ** 2) * (2.0 * dt / (n * pcg))
    P[0] /= 2
    if n % 2 == 0 and len(P) > 1:
        P[-1] /= 2
    return np.fft.rfftfreq(n, d=dt), P


def cumulative_power(P):
    """パワーの累積比率（0→1）を返す。"""
    c = np.cumsum(P)
    total = c[-1]
    return c / total if total > 0 else np.zeros_like(c)


def power_quantile_freq(f, P, q):
    """累積パワーが比率 q に達する周波数を返す。"""
    c = cumulative_power(P)
    i = int(np.searchsorted(c, q))
    return float(f[min(i, len(f) - 1)])


def log_decimate(f, v, n_per_decade=60, how="mean"):
    """全帯域スペクトルを対数間引きして軽量化する（描画用キャッシュ）。

    NR-500 は rfft 後 500,001点になるため、そのまま保持すると重い。
    対数間引きなら数百点で log-log グラフの見た目を保てる。
    """
    f = np.asarray(f, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    m = f > 0
    f, v = f[m], v[m]
    if len(f) < 2:
        return f, v

    n_bins = max(int(np.ceil(np.log10(f[-1] / f[0]) * n_per_decade)), 2)
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), n_bins + 1)
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bins - 1)

    cnt = np.bincount(idx, minlength=n_bins)
    if how == "max":
        agg = np.full(n_bins, -np.inf)
        np.maximum.at(agg, idx, v)
    else:
        agg = np.bincount(idx, weights=v, minlength=n_bins) / np.where(cnt > 0, cnt, 1)

    keep = cnt > 0
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers[keep], agg[keep]


def lowpass_fft(y, dt, fc, roll=0.2):
    """ゼロ位相のFFTローパス（カットオフ掃引の評価用）。

    fc から fc*(1+roll) へ cos^2 でなだらかに落とす。
    ※これは評価用。実際にファイルへ適用する処理は本スコープ外
      （scipy.signal.butter + filtfilt を使うのが適切）。
    """
    n = len(y)
    F = np.fft.rfft(np.asarray(y, dtype=np.float64))
    f = np.fft.rfftfreq(n, d=dt)
    H = np.ones_like(f)
    f2 = fc * (1.0 + roll)
    band = (f > fc) & (f < f2)
    H[band] = np.cos(np.pi / 2 * (f[band] - fc) / (f2 - fc)) ** 2
    H[f >= f2] = 0.0
    return np.fft.irfft(F * H, n=n)


# ==========================================================
# ファイル1本ぶんの指標
# ==========================================================
def robust_sigma(y):
    """MADベースのばらつき推定。MADが0になる場合は std にフォールバックする。

    Futaba形式は 0.01MPa 刻みに量子化されており、静穏部では過半のサンプルが
    同じ値になることがある。そのとき MAD が 0 になってしまうので、
    フォールバックがないとノイズ量を 0 と誤判定する。
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    mad = float(np.median(np.abs(y - np.median(y))))
    return 1.4826 * mad if mad > 0 else float(np.std(y))


def _baseline_noise(y, i0, i1, fallback, min_points=50):
    """イベント前（なければイベント後）の静穏部からノイズ量を robust に見積もる。

    単純な std だと、イベント後に残る圧力プラトーを拾って過大評価になる。
    MAD ベースにすることで、多少の段差や外れ値があっても影響を受けにくい。
    """
    for seg in (y[:i0], y[i1 + 1:]):
        if len(seg) >= min_points:
            return robust_sigma(seg)
    return float(fallback)


def metrics(y, dt, use_event=True, detrend=True, window="hann", zero_rows=None):
    """1チャンネルぶんの解析指標をまとめて返す。

    zero_rows を渡すとゼロ点合わせ窓の汚染検査も行い、フラグに反映する。
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    flags = []

    i0, i1, ev_flags, info = find_event(y, dt)
    flags += ev_flags
    if not use_event:
        i0, i1 = 0, n - 1

    if zero_rows:
        zc = zero_adjust_check(y, zero_rows)
        flags += zc["flags"]
    else:
        zc = {}

    seg = y[i0:i1 + 1]
    f, P = psd(seg, dt, detrend=detrend, window=window)

    base = info["baseline"]
    # ノイズはイベント「前」の静穏部から測る。
    # イベント後は残圧のプラトーが残ることがあり、そこを含めると過大評価になるため。
    # 前が取れない（イベントが記録先頭にある）場合はイベント後にフォールバックする。
    noise_rms = _baseline_noise(y, i0, i1, fallback=info["noise_sigma"])

    tr = rise_time(y, dt, i0, i1, baseline=base)
    out = {
        "n_total": n,
        "dt": dt,
        "duration_s": n * dt,
        "baseline_MPa": base,
        "noise_sigma_MPa": info["noise_sigma"],
        "noise_rms_MPa": noise_rms,
        "peak_MPa": info["peak"],
        "peak_minus_baseline_MPa": info["amp"],
        "snr": info["snr"],
        "event_start_s": info["event_start"],
        "event_end_s": info["event_end"],
        "event_width_s": info["event_width"],
        "analysis_start_s": i0 * dt,
        "analysis_end_s": i1 * dt,
        "analysis_points": int(i1 - i0 + 1),
        "rise_time_ms": tr * 1000.0 if np.isfinite(tr) else float("nan"),
        "required_bw_Hz": (0.35 / tr) if (np.isfinite(tr) and tr > 0) else float("nan"),
        "freq_resolution_Hz": 1.0 / ((i1 - i0 + 1) * dt),
        "nyquist_Hz": 1.0 / (2 * dt),
        "f90_Hz": power_quantile_freq(f, P, 0.90),
        "f95_Hz": power_quantile_freq(f, P, 0.95),
        "f99_Hz": power_quantile_freq(f, P, 0.99),
        "f999_Hz": power_quantile_freq(f, P, 0.999),
        "flags": ",".join(flags) if flags else "ok",
    }
    out.update({k: v for k, v in zc.items() if k != "flags"})
    return out, (f, P), (i0, i1)


# ==========================================================
# 出力ファイル名のユーティリティ
# ==========================================================
def folder_tag(dirpath, main_dir):
    """main_dir からの相対パスをファイル名用のタグにする。

    末端フォルダ名だけだと 0.5mm/190℃ と 1.0mm/190℃ が同名になり
    上書きされてしまうため、相対パス全体を使って衝突を防ぐ。
    """
    rel = os.path.relpath(dirpath, main_dir)
    return "root" if rel == "." else rel.replace(os.sep, "_")


def list_csv_by_format(main_dir):
    """フォルダ配下のCSVを形式ごとに仕分けして返す。

    戻り値: {"futaba": [...], "nr500": [...], "other": [...]}
    """
    buckets = {"futaba": [], "nr500": [], "other": []}
    for dirpath, _dirnames, filenames in os.walk(main_dir):
        for name in sorted(filenames):
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(dirpath, name)
            fmt = detect_format(path)
            buckets[fmt if fmt else "other"].append(path)
    return buckets


def setup_japanese_font(plt):
    """グラフの日本語文字化け対策（Windows最適化）。"""
    import warnings
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message=".*findfont: Font family.*not found.*"
    )


# ==========================================================
# 共通周波数グリッド（複数ファイルのパーセンタイル集計用）
# ==========================================================
def make_log_grid(f_min=0.1, f_max=1.0e5, n_per_decade=60):
    """全ファイル共通の対数周波数グリッド（ビン端とビン中心）を作る。

    Futaba（Nyquist 500Hz）と NR-500（Nyquist 100kHz）でスペクトルの
    長さも分解能も違うため、同じグリッドに載せないとパーセンタイルが取れない。
    """
    n = max(int(round(np.log10(f_max / f_min) * n_per_decade)), 2)
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def bin_to_grid(f, v, edges):
    """スペクトルを指定の対数ビンへ平均で畳み込む。空のビンは NaN。

    そのファイルの帯域外（Nyquistより上、周波数分解能より下）は NaN になるので、
    np.nanpercentile で自然に「寄与できるファイルだけ」で集計できる。
    """
    f = np.asarray(f, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    m = (f >= edges[0]) & (f <= edges[-1]) & (f > 0)
    f, v = f[m], v[m]
    n_bins = len(edges) - 1
    out = np.full(n_bins, np.nan)
    if len(f) == 0:
        return out
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    tot = np.bincount(idx, weights=v, minlength=n_bins)
    nz = cnt > 0
    out[nz] = tot[nz] / cnt[nz]

    # 低周波側では対数ビンの幅が元スペクトルの分解能より狭くなり空ビンが生じる。
    # そのファイルが実際にカバーしている帯域内の空ビンだけを補間で埋め、
    # 帯域外（Nyquistより上など）は NaN のまま残す。
    centers = np.sqrt(edges[:-1] * edges[1:])
    inside = (centers >= f[0]) & (centers <= f[-1])
    known = np.isfinite(out)
    need = inside & ~known
    if need.any() and known.sum() >= 2:
        out[need] = np.interp(np.log10(centers[need]),
                              np.log10(centers[known]), out[known])
    return out
