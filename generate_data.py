"""
Phase 1: Synthetic Dataset Generator
Generates 10,000 B2B SaaS customer records with intentional Paid Social underperformance
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 10_000

# ── Channel config: Paid Social intentionally underperforms ──────────────────
CHANNEL_CONFIG = {
    "Paid Social": {
        "weight": 0.30,
        "cac_mean": 620,  "cac_std": 180,   # high CAC
        "arpu_mean": 310, "arpu_std": 90,    # low ARPU
        "churn_mean": 5,  "churn_std": 1.5,  # high churn (short tenure)
    },
    "Organic SEO": {
        "weight": 0.28,
        "cac_mean": 190,  "cac_std": 60,
        "arpu_mean": 480, "arpu_std": 110,
        "churn_mean": 18, "churn_std": 5,
    },
    "Email": {
        "weight": 0.22,
        "cac_mean": 140,  "cac_std": 45,
        "arpu_mean": 420, "arpu_std": 95,
        "churn_mean": 16, "churn_std": 4,
    },
    "Referral": {
        "weight": 0.20,
        "cac_mean": 110,  "cac_std": 35,
        "arpu_mean": 560, "arpu_std": 130,
        "churn_mean": 22, "churn_std": 6,
    },
}

SEGMENT_CONFIG = {
    "SMB":        {"weight": 0.65, "arpu_mult": 0.75, "cac_mult": 0.80},
    "Enterprise": {"weight": 0.35, "arpu_mult": 1.60, "cac_mult": 1.45},
}

channels  = list(CHANNEL_CONFIG.keys())
ch_weights = [CHANNEL_CONFIG[c]["weight"] for c in channels]
segments  = list(SEGMENT_CONFIG.keys())
seg_weights = [SEGMENT_CONFIG[s]["weight"] for s in segments]

chosen_channels = np.random.choice(channels,  size=N, p=ch_weights)
chosen_segments = np.random.choice(segments, size=N, p=seg_weights)

cac_list, arpu_list, churn_list = [], [], []

for ch, seg in zip(chosen_channels, chosen_segments):
    cc = CHANNEL_CONFIG[ch]
    sc = SEGMENT_CONFIG[seg]

    cac  = max(50,  np.random.normal(cc["cac_mean"]  * sc["cac_mult"],  cc["cac_std"]))
    arpu = max(50,  np.random.normal(cc["arpu_mean"] * sc["arpu_mult"], cc["arpu_std"]))
    churn_month = max(1, int(np.random.normal(cc["churn_mean"], cc["churn_std"])))

    cac_list.append(round(cac, 2))
    arpu_list.append(round(arpu, 2))
    churn_list.append(churn_month)

df = pd.DataFrame({
    "Customer_ID":         [f"CUST_{i:05d}" for i in range(1, N + 1)],
    "Acquisition_Channel": chosen_channels,
    "Segment":             chosen_segments,
    "CAC":                 cac_list,
    "Monthly_Revenue":     arpu_list,
    "Churn_Month":         churn_list,
})

out = "saas_cac_ltv_data.csv"
df.to_csv(out, index=False)
print(f"✅  Saved {len(df):,} records → {out}")
print(df.groupby("Acquisition_Channel")[["CAC","Monthly_Revenue","Churn_Month"]].mean().round(1))
