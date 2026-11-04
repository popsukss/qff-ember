"""Coverage + label summary per station for the 1 May - 30 Sep fire season."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/raw/stations.csv", skipinitialspace=True)
df["date"] = pd.to_datetime(df["rep_date"]).dt.normalize()
df["name"] = df["name"].str.strip()
df = df[(df.date.dt.month >= 5) & (df.date.dt.month <= 9)]
df["year"] = df.date.dt.year

for aes, g in df.groupby("aes"):
    g = g.drop_duplicates("date")
    per = g.groupby("year").agg(days=("date", "size"), fwi_ok=("fwi", lambda s: s.notna().sum()))
    per["ok_frac"] = per.fwi_ok / 153
    good = per[per.ok_frac >= 0.9]
    print(f"\n=== {g['name'].iloc[0]} (aes {aes})  years {per.index.min()}-{per.index.max()}  "
          f"seasons={len(per)}  seasons>=90% complete={len(good)}")
    print("incomplete seasons:", {int(y): round(f, 2) for y, f in per.ok_frac.items() if f < 0.9})
    tr = g[(g.year >= 2000) & (g.year <= 2017)].fwi.dropna()
    q95 = tr.quantile(0.95)
    print(f"train FWI: n={len(tr)} mean={tr.mean():.1f} median={tr.median():.1f} q95={q95:.1f} max={tr.max():.1f}")
    for split, (a, b) in {"val": (2018, 2021), "test": (2022, 2026)}.items():
        s = g[(g.year >= a) & (g.year <= b)].fwi.dropna()
        print(f"{split}: n={len(s)} extreme days (>=q95)={int((s >= q95).sum())}")
    y23 = g[g.year == 2023].set_index("date").fwi
    if len(y23):
        top = y23.nlargest(5)
        print("2023 top FWI days:", {d.strftime('%m-%d'): round(v, 1) for d, v in top.items()})
        print("2023 extreme days:", int((y23 >= q95).sum()), " Aug:", int((y23[y23.index.month == 8] >= q95).sum()))
