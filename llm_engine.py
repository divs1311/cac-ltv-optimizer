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

Output format — always respond in valid JSON with this exact schema:
{
  "executive_summary": "2-3 sentence headline finding",
  "critical_flags": ["flag1", "flag2"],
  "channel_recommendations": [
    {
      "channel": "Channel Name",
      "action": "REALLOCATE | SCALE | OPTIMIZE | SUNSET",
      "budget_shift_pct": 15,
      "rationale": "data-backed reason",
      "expected_payback_compression_months": 2.5,
      "ltv_expansion_tactic": "specific tactic"
    }
  ],
  "ltv_expansion_plays": ["play1", "play2", "play3"],
  "30_day_quick_wins": ["win1", "win2"],
  "90_day_strategic_moves": ["move1", "move2"],
  "projected_blended_roi_lift_pct": 25
}

Return ONLY the JSON object. No markdown, no backticks, no explanation."""


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
            flags.append(f"🔴 HIGH_CAC_FLAG: {name} — CAC:LTV = {ratio:.2f} (threshold: 0.33)")
        if payback > 12:
            flags.append(f"🔴 PAYBACK_BREACH: {name} — Payback = {payback:.1f} months (SaaS benchmark: <12)")
        if high_cac_pct > 0.50:
            flags.append(f"🟡 COHORT_RISK: {name} — {high_cac_pct:.0%} of customers in high-CAC zone")

    return "\n".join(flags) if flags else "No critical flags."


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_prompt(channel_summary: list, cohort_summary: list, total_budget: float = 500_000) -> str:
    """Assembles data tables and diagnostic flags into a structured prompt."""
    return f"""
UNIT ECONOMICS DIAGNOSTIC REPORT — B2B SaaS Fintech Platform
=============================================================
Total Monthly Acquisition Budget: ${total_budget:,.0f}
Analysis Period: Current cohort snapshot (10,000 customers)

CHANNEL-LEVEL PERFORMANCE:
{json.dumps(channel_summary, indent=2)}

COHORT-LEVEL BREAKDOWN (Channel x Segment):
{json.dumps(cohort_summary, indent=2)}

RULE-BASED FLAGS TRIGGERED:
{_build_flags(channel_summary)}

TASK:
1. Identify which channels are destroying unit economics (CAC:LTV > 0.33 = critical)
2. Build a specific budget reallocation plan with % shifts
3. Prescribe LTV expansion tactics for underperforming cohorts (focus on churn reduction)
4. Compress the blended Payback Period across all channels
5. Align recommendations to a cross-border B2B payments platform growth model

Be specific with numbers. Cite the CAC, LTV, and Payback Period in your rationale.
Return ONLY valid JSON matching the schema. No extra text.
"""


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

    # Configure Gemini SDK
    genai.configure(api_key=api_key)

    # Build prompt
    prompt = build_prompt(channel_summary, cohort_summary, total_budget)

    # Initialize model with system instruction
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    # Call Gemini async
    response = await model.generate_content_async(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 1500,
            "response_mime_type": "application/json",
        },
    )

    # Extract and clean response text
    raw = response.text
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    return json.loads(clean)


# ── Sync Wrapper for Streamlit ─────────────────────────────────────────────────
def get_ai_recommendations_sync(
    channel_summary: list,
    cohort_summary: list,
    total_budget: float = 500_000,
) -> dict:
    """Synchronous wrapper — Streamlit cannot call async functions directly."""
    return asyncio.run(get_ai_recommendations(channel_summary, cohort_summary, total_budget))