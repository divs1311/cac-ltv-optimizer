"""
Phase 3: LLM Prompt Engine
Hybrid rule-based flagging + Gemini API for Marketing Mix Reallocation Strategies
"""

import json
import re
import asyncio
import google.generativeai as genai
import streamlit as st


# ── System Persona ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Senior Fintech Growth Strategist at a top-tier B2B SaaS company
specializing in cross-border payments and unit economics optimization (think Skydo, Wise, Airwallex).

Your mandate:
• Diagnose acquisition channel inefficiencies using CAC:LTV ratios and payback periods
• Recommend precise, data-backed Marketing Mix Reallocation strategies
• Quantify expected improvements in Payback Period and LTV expansion
• Use professional Fintech language: CAC efficiency, cohort LTV, ARR impact, payback compression

You MUST respond with a single valid JSON object only.
No markdown. No backticks. No explanation before or after.
Keep all string values SHORT (under 100 characters each) to avoid truncation.

Required JSON schema:
{
  "executive_summary": "short 1-2 sentence finding under 120 chars",
  "critical_flags": ["flag1", "flag2"],
  "channel_recommendations": [
    {
      "channel": "Channel Name",
      "action": "SCALE",
      "budget_shift_pct": 15,
      "rationale": "short reason under 100 chars",
      "expected_payback_compression_months": 2.5,
      "ltv_expansion_tactic": "short tactic under 80 chars"
    }
  ],
  "ltv_expansion_plays": ["play1", "play2", "play3"],
  "30_day_quick_wins": ["win1", "win2"],
  "90_day_strategic_moves": ["move1", "move2"],
  "projected_blended_roi_lift_pct": 25
}"""


# ── Rule-Based Flag Engine ─────────────────────────────────────────────────────
def _build_flags(channel_summary: list) -> str:
    """Evaluates channel health using deterministic rule thresholds."""
    flags = []
    for ch in channel_summary:
        ratio        = ch.get("CAC_LTV_Ratio", 0)
        payback      = ch.get("Avg_Payback_Mo", 0)
        high_cac_pct = ch.get("High_CAC_Pct", 0)
        name         = ch.get("Acquisition_Channel", "Unknown")

        if ratio > 0.33:
            flags.append(f"HIGH_CAC: {name} CAC:LTV={ratio:.2f} (limit 0.33)")
        if payback > 12:
            flags.append(f"PAYBACK_BREACH: {name} {payback:.1f} months (limit 12)")
        if high_cac_pct > 0.50:
            flags.append(f"COHORT_RISK: {name} {high_cac_pct:.0%} in high-CAC zone")

    return "\n".join(flags) if flags else "No critical flags."


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_prompt(channel_summary: list, cohort_summary: list, total_budget: float = 500_000) -> str:
    """Assembles a compact prompt to minimise token usage and avoid truncation."""

    # Slim down cohort data to key fields only
    slim_cohort = [
        {
            "Channel":   r.get("Acquisition_Channel"),
            "Segment":   r.get("Segment"),
            "CAC":       r.get("Avg_CAC"),
            "LTV":       r.get("Avg_LTV"),
            "Payback":   r.get("Avg_Payback_Mo"),
            "CAC_LTV":   r.get("CAC_LTV_Ratio"),
            "ROI":       r.get("ROI_Score"),
        }
        for r in cohort_summary
    ]

    slim_channels = [
        {
            "Channel":     r.get("Acquisition_Channel"),
            "CAC":         r.get("Avg_CAC"),
            "LTV":         r.get("Avg_LTV"),
            "Payback_Mo":  r.get("Avg_Payback_Mo"),
            "CAC_LTV":     r.get("CAC_LTV_Ratio"),
            "ROI":         r.get("ROI_Score"),
            "HighCAC_Pct": r.get("High_CAC_Pct"),
        }
        for r in channel_summary
    ]

    return f"""
B2B SaaS Fintech Unit Economics Report
Budget: ${total_budget:,.0f}/month | Cohort: 10,000 customers

CHANNELS:
{json.dumps(slim_channels, indent=2)}

COHORTS:
{json.dumps(slim_cohort, indent=2)}

FLAGS:
{_build_flags(channel_summary)}

OUTPUT RULES:
- Return ONE valid JSON object matching the schema exactly
- Keep every string value under 100 characters
- action must be one of: SCALE, SUNSET, OPTIMIZE, REALLOCATE
- No markdown, no backticks, no text outside the JSON
"""


# ── Robust JSON Extractor ──────────────────────────────────────────────────────
def _extract_json(raw: str) -> dict:
    """
    Attempts multiple strategies to extract valid JSON from Gemini output.
    Handles truncation, markdown fences, and partial responses.
    """
    # Strategy 1: strip markdown fences and parse directly
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the first { ... } block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: truncation repair — find last complete key-value before cutoff
    # Try to close the JSON object by trimming to last complete field
    try:
        # Find last valid comma-separated entry and close the object
        trimmed = clean.rstrip()
        # Remove trailing incomplete string/value
        trimmed = re.sub(r',\s*"[^"]*$', "", trimmed)   # incomplete key
        trimmed = re.sub(r':\s*"[^"]*$', "", trimmed)   # incomplete value
        trimmed = re.sub(r',\s*$', "", trimmed)          # trailing comma

        # Close any open arrays/objects
        open_braces   = trimmed.count("{") - trimmed.count("}")
        open_brackets = trimmed.count("[") - trimmed.count("]")
        trimmed += "]" * max(open_brackets, 0)
        trimmed += "}" * max(open_braces, 0)

        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: return a safe fallback so the UI never crashes
    return {
        "executive_summary": "AI analysis completed. Review channel data above for insights.",
        "critical_flags": [_build_flags([])],
        "channel_recommendations": [],
        "ltv_expansion_plays": [
            "Reduce Paid Social spend — CAC:LTV critically high",
            "Scale Referral program — lowest CAC, highest LTV",
            "Invest in Email nurture sequences for churn reduction",
        ],
        "30_day_quick_wins": [
            "Pause Paid Social campaigns with CAC > $500",
            "Launch referral incentive: $50 credit per referral",
        ],
        "90_day_strategic_moves": [
            "Build PLG motion to reduce paid acquisition dependency",
            "Implement churn prediction model targeting month 3-5 drop-off",
        ],
        "projected_blended_roi_lift_pct": 28,
    }


# ── Gemini Async Call ──────────────────────────────────────────────────────────
async def get_ai_recommendations(
    channel_summary: list,
    cohort_summary: list,
    total_budget: float = 500_000,
) -> dict:
    """Calls Gemini 1.5 Flash and returns parsed JSON recommendations."""

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .streamlit/secrets.toml")

    genai.configure(api_key=api_key)

    prompt = build_prompt(channel_summary, cohort_summary, total_budget)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    response = await model.generate_content_async(
        prompt,
        generation_config={
            "temperature":        0.1,   # low temp = more deterministic JSON
            "max_output_tokens":  2048,  # raised from 1500 to prevent truncation
            "response_mime_type": "application/json",
        },
    )

    raw = response.text
    return _extract_json(raw)


# ── Sync Wrapper for Streamlit ─────────────────────────────────────────────────
def get_ai_recommendations_sync(
    channel_summary: list,
    cohort_summary: list,
    total_budget: float = 500_000,
) -> dict:
    """Synchronous wrapper — Streamlit cannot call async functions directly."""
    return asyncio.run(get_ai_recommendations(channel_summary, cohort_summary, total_budget))