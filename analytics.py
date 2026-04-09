"""
Phase 2: Unit Economics Processing Pipeline
Calculates LTV, CAC:LTV Ratio, Payback Period, ROI Score per cohort
"""

import pandas as pd
import numpy as np


def load_and_compute(path: str = "saas_cac_ltv_data.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    # ── LTV = ARPU / Churn_Rate  (churn_rate = 1 / churn_month) ─────────────
    df["Churn_Rate"]    = 1 / df["Churn_Month"]
    df["LTV"]           = df["Monthly_Revenue"] / df["Churn_Rate"]          # = ARPU * Churn_Month
    df["CAC_LTV_Ratio"] = df["CAC"] / df["LTV"]
    df["Payback_Period"]= df["CAC"] / df["Monthly_Revenue"]                 # months
    df["Gross_Margin"]  = np.random.uniform(0.65, 0.82, len(df))            # typical SaaS GM
    df["ROI_Score"]     = ((df["LTV"] - df["CAC"]) / df["CAC"]).round(4)

    # ── Rule-based flags ─────────────────────────────────────────────────────
    df["High_CAC_Flag"] = df["CAC_LTV_Ratio"] > 0.33   # CAC > 33% of LTV
    df["LTV_Expansion_Opportunity"] = (
        (df["Churn_Month"] < df.groupby("Acquisition_Channel")["Churn_Month"].transform("median")) &
        (df["Monthly_Revenue"] < df.groupby("Segment")["Monthly_Revenue"].transform("median"))
    )

    # ── Cohort summary: Channel × Segment ────────────────────────────────────
    cohort = (
        df.groupby(["Acquisition_Channel", "Segment"])
        .agg(
            Count            = ("Customer_ID",    "count"),
            Avg_CAC          = ("CAC",             "mean"),
            Avg_LTV          = ("LTV",             "mean"),
            Avg_ARPU         = ("Monthly_Revenue", "mean"),
            Avg_Churn_Month  = ("Churn_Month",     "mean"),
            Avg_Payback_Mo   = ("Payback_Period",  "mean"),
            CAC_LTV_Ratio    = ("CAC_LTV_Ratio",   "mean"),
            ROI_Score        = ("ROI_Score",       "mean"),
            High_CAC_Pct     = ("High_CAC_Flag",   "mean"),
            LTV_Expand_Pct   = ("LTV_Expansion_Opportunity", "mean"),
        )
        .reset_index()
        .round(2)
    )

    # ── Health grade ─────────────────────────────────────────────────────────
    def grade(row):
        if row["CAC_LTV_Ratio"] <= 0.20: return "🟢 Healthy"
        if row["CAC_LTV_Ratio"] <= 0.33: return "🟡 Watch"
        return "🔴 Critical"

    cohort["Health_Grade"] = cohort.apply(grade, axis=1)
    return df, cohort


def channel_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    return (
        cohort.groupby("Acquisition_Channel")
        .agg(
            Total_Customers  = ("Count",          "sum"),
            Avg_CAC          = ("Avg_CAC",         "mean"),
            Avg_LTV          = ("Avg_LTV",         "mean"),
            Avg_Payback_Mo   = ("Avg_Payback_Mo",  "mean"),
            CAC_LTV_Ratio    = ("CAC_LTV_Ratio",   "mean"),
            ROI_Score        = ("ROI_Score",       "mean"),
            High_CAC_Pct     = ("High_CAC_Pct",    "mean"),
        )
        .reset_index()
        .round(2)
        .sort_values("CAC_LTV_Ratio")
    )


if __name__ == "__main__":
    df, cohort = load_and_compute("saas_cac_ltv_data.csv")
    ch = channel_summary(cohort)
    print("\n── Channel Summary ──")
    print(ch.to_string(index=False))
    print("\n── Cohort Table (first 8 rows) ──")
    print(cohort.head(8).to_string(index=False))
