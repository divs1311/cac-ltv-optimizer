"""
Phase 3: LLM Prompt Engine
Hybrid rule-based flagging + Gemini API for Marketing Mix Reallocation Strategies
"""
 
import json
import re
import httpx
 
 
# ── System persona ────────────────────────────────────────────────────────────
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
import google.generativeai as genai
import streamlit as st

# This safely pulls the string from your hidden local secrets.toml
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def run_ai_analysis(your_prompt):
    try:
        # Using a stable production model release
        # Change "gemini-1.5-flash" to "gemini-2.5-flash" or "gemini-1.5-flash-002"
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(your_prompt)
        return response.text
    except Exception as e:
        st.error(f"AI Analysis Error: {e}")
        return None
 
def _build_flags(channel_summary: list) -> str:
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
 
 
def build_prompt(channel_summary: list, cohort_summary: list, total_budget: float = 500_000) -> str:
    return f"""
UNIT ECONOMICS DIAGNOSTIC REPORT — B2B SaaS Fintech Platform
=============================================================
Total Monthly Acquisition Budget: ${total_budget:,.0f}
Analysis Period: Current cohort snapshot (10,000 customers)
 
CHANNEL-LEVEL PERFORMANCE:
{json.dumps(channel_summary, indent=2)}
 
COHORT-LEVEL BREAKDOWN (Channel × Segment):
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
 
 
async def get_ai_recommendations(
    channel_summary: list,
    cohort_summary: list,
    total_budget: float = 500_000,
) -> dict:
    """Call Gemini API and return parsed JSON recommendations."""
 
    import streamlit as st
    api_key = st.secrets.get("GEMINI_API_KEY", "")
 
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in secrets.toml")
 
    prompt = build_prompt(channel_summary, cohort_summary, total_budget)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
 
    # Gemini 1.5 Flash endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
 
    payload = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1500,
        }
    }
 
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        data = response.json()
 
    # Parse Gemini response format
    if "candidates" not in data:
        raise ValueError(f"Unexpected Gemini response: {data}")
 
    raw = data["candidates"][0]["content"]["parts"][0]["text"]
 
    # Strip markdown fences if Gemini adds them
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)
 
 
def get_ai_recommendations_sync(
    channel_summary: list,
    cohort_summary: list,
    total_budget: float = 500_000,
) -> dict:
    """Synchronous wrapper used by Streamlit."""
    import asyncio
    return asyncio.run(get_ai_recommendations(channel_summary, cohort_summary, total_budget))