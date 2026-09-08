# -*- coding: utf-8 -*-
"""00_コード説明.md の記載が実物のノートブックと一致するか照合する

実行方法（プロジェクトルートから）:
    .venv/Scripts/python.exe 検証/test_docs.py
"""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "コードフォルダ")
DOC = os.path.join(CODE, "00_コード説明.md")

_fails = []


def check(label, ok, detail=""):
    print(f"  {'OK ' if ok else 'NG '} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _fails.append(label)


def cells(key):
    name = next(n for n in sorted(os.listdir(CODE))
                if n.startswith(key) and n.endswith(".ipynb"))
    nb = json.load(io.open(os.path.join(CODE, name), encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


doc = io.open(DOC, encoding="utf-8").read()
KEYS = ["01", "02", "03", "04", "05", "06", "07"]

print("[1] セル数（ドキュメント記載と一致するか）")
for key, n in zip(KEYS, [2, 4, 8, 2, 4, 8, 9]):
    cs = cells(key)
    check(f"{key}: {n}セル", len(cs) == n, f"(実際 {len(cs)})")

print("\n[2] 設定変数の場所")
loc = {
    ("01", 0): ["ZERO_ADJUST", "USE_EVENT_WINDOW", "DETREND", "WINDOW"],
    ("01", 1): ["LOG_Y", "X_MIN", "X_MAX", "X_STEP", "Y_MIN", "Y_MAX"],
    ("02", 0): ["ZERO_ADJUST", "USE_EVENT_WINDOW", "DETREND", "WINDOW"],
    ("02", 3): ["LOG_Y", "X_MIN", "X_MAX", "X_STEP"],
    ("03", 0): ["ZERO_ADJUST", "USE_EVENT_WINDOW", "DETREND", "WINDOW"],
    ("03", 3): ["LOG_Y", "X_MIN", "X_MAX", "X_STEP", "CACHE_MAX_HZ"],
    ("04", 0): ["ZERO_ADJUST", "USE_EVENT_WINDOW", "DETREND", "WINDOW"],
    ("04", 1): ["LOG_Y", "X_MIN", "X_MAX", "X_STEP"],
    ("05", 0): ["ZERO_ADJUST", "USE_EVENT_WINDOW", "DETREND", "WINDOW"],
    ("05", 3): ["LOG_Y", "X_MIN", "X_MAX", "X_STEP"],
    ("06", 0): ["ZERO_ADJUST", "USE_EVENT_WINDOW", "DETREND", "WINDOW"],
    ("06", 3): ["LOG_Y", "X_MIN", "X_MAX", "X_STEP", "CACHE_MAX_HZ"],
    ("07", 3): ["USE_EVENT_WINDOW", "DETREND", "WINDOW", "CUTOFF_CANDIDATES"],
}
for (key, idx), names in loc.items():
    src = cells(key)[idx]
    missing = [n for n in names if f"{n} " not in src and f"{n}," not in src
               and f"{n}=" not in src]
    check(f"{key} セル{idx + 1}: {', '.join(names[:3])}...", not missing,
          f"(不足 {missing})" if missing else "")

print("\n[3] ゼロ点合わせの既定値（Futaba=False / NR-500=True）")
for key, want in (("01", "False"), ("02", "False"), ("03", "False"),
                  ("04", "True"), ("05", "True"), ("06", "True")):
    src = cells(key)[0]
    check(f"{key}: ZERO_ADJUST = {want}", f"ZERO_ADJUST = {want}" in src)

print("\n[4] チャンネル定義")
for key, want in (("01", '["CH03", "CH04"]'), ("02", '["CH03", "CH04"]'),
                  ("03", '["CH03", "CH04"]'), ("04", '["V03", "V04"]'),
                  ("05", '["V03", "V04"]'), ("06", '["V03", "V04"]')):
    check(f"{key}: CHANNELS = {want}", f"CHANNELS = {want}" in cells(key)[0])

print("\n[5] 出力フォルダ名（ドキュメント記載と実物）")
outs = {"02": "fft_results", "03": "fft_results_per_channel",
        "05": "fft_results_nr500", "06": "fft_results_per_channel_nr500",
        "07": "fft_cutoff_analysis"}
for key, d in outs.items():
    joined = "\n".join(cells(key))
    check(f"{key}: {d}", f'"{d}"' in joined and d in doc)

print("\n[6] 出力ファイル名のパターン")
check("02/05: [相対パス]_[CSV名]_fft.png",
      all("_fft.png" in "\n".join(cells(k)) for k in ("02", "05")))
check("03/06: _combined_fft.png / _ALL_combined_fft.png",
      all("_ALL_combined_fft.png" in "\n".join(cells(k)) for k in ("03", "06")))
check("02/03/05/06: metrics_per_file.csv",
      all("metrics_per_file.csv" in "\n".join(cells(k))
          for k in ("02", "03", "05", "06")))
check("07: cutoff_sweep.csv / cutoff_summary.csv",
      all(n in "\n".join(cells("07"))
          for n in ("cutoff_sweep.csv", "cutoff_summary.csv")))

print("\n[7] 全7本が fft_common を読み込んでいる")
for key in KEYS:
    src = cells(key)[0]
    check(f"{key}: fft_common をロード",
          "fft_common.py" in src and "from fft_common import" in src)

print("\n[8] 共通モジュールの定数")
mod = io.open(os.path.join(CODE, "fft_common.py"), encoding="utf-8").read()
check("VOLT_TO_MPA = 20.0", "VOLT_TO_MPA = 20.0" in mod)
check("ZERO_ADJUST_ROWS = 1000", "ZERO_ADJUST_ROWS = 1000" in mod)
check("ドキュメントに 圧力[MPa] = 電圧[V] × 20 の記載", "電圧[V] × 20" in doc)
check("ドキュメントに 最初の 1000 行 の記載", "1000 行の平均" in doc)

print("\n[9] 品質フラグがドキュメントと実装で一致")
for flag in ("no_event", "event_at_edge", "multiple_events", "low_snr",
             "short_event", "zero_window_offset", "zero_window_noisy"):
    check(f"{flag}", flag in mod and flag in doc)

print("\n" + "=" * 60)
if _fails:
    print(f"NG {len(_fails)} 件:")
    for f in _fails:
        print("  -", f)
    sys.exit(1)
print("全項目 OK")
