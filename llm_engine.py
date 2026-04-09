"""
Phase 3: LLM Prompt Engine
Hybrid rule-based flagging + Multi-LLM API support (OpenAI, Gemini, Anthropic)
"""

import json
import re
import httpx
import os


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
}"""


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(channel_summary: dict, cohort_summary: dict, total_budget: float = 500_000) -> str:
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
"""


# ── Rule-Based Flags ──────────────────────────────────────────────────────────
def _build_flags(channel_summary: dict) -> str:
    flags = []
    for ch in channel_summary:
        ratio = ch.get("CAC_LTV_Ratio", 0)
        payback = ch.get("Avg_Payback_Mo", 0)
        high_cac_pct = ch.get("High_CAC_Pct", 0)
        name = ch.get("Acquisition_Channel", "Unknown")

        if ratio > 0.33:
            flags.append(f"🔴 HIGH_CAC_FLAG: {name} — CAC:LTV = {ratio:.2f} (threshold: 0.33)")
        if payback > 12:
            flags.append(f"🔴 PAYBACK_BREACH: {name} — Payback = {payback:.1f} months (SaaS benchmark: <12)")
        if high_cac_pct > 0.50:
            flags.append(f"🟡 COHORT_RISK: {name} — {high_cac_pct:.0%} of customers in high-CAC zone")

    return "\n".join(flags) if flags else "No critical flags."


# ── API Key Loader ────────────────────────────────────────────────────────────
def _get_api_key(provider: str) -> str:
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "google": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }

    api_key = keys.get(provider)
    if not api_key:
        raise ValueError(f"Missing API key for provider: {provider}")

    return api_key


# ── Provider Handlers ─────────────────────────────────────────────────────────
async def _call_openai(client, api_key, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    response = await client.post(url, headers=headers, json=payload)
    data = response.json()
    return data['choices'][0]['message']['content']


async def _call_gemini(client, api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]
        }],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    response = await client.post(url, json=payload)
    data = response.json()
    return data['candidates'][0]['content']['parts'][0]['text']


async def _call_anthropic(client, api_key, prompt):
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = await client.post(url, headers=headers, json=payload)
    data = response.json()
    return data['content'][0]['text']


# ── Provider Router ───────────────────────────────────────────────────────────
_PROVIDER_MAP = {
    "openai": _call_openai,
    "google": _call_gemini,
    "anthropic": _call_anthropic
}


# ── Main Engine ───────────────────────────────────────────────────────────────
async def get_ai_recommendations(
    channel_summary: list[dict],
    cohort_summary: list[dict],
    total_budget: float = 500_000,
) -> dict:

    prompt = build_prompt(channel_summary, cohort_summary, total_budget)

    # Change provider here OR set via env variable
    provider = os.getenv("LLM_PROVIDER", "openai")

    api_key = _get_api_key(provider)

    async with httpx.AsyncClient(timeout=60) as client:
        handler = _PROVIDER_MAP.get(provider)

        if not handler:
            raise ValueError(f"Unsupported provider: {provider}")

        raw = await handler(client, api_key, prompt)

    # ── Clean + Parse JSON ────────────────────────────────────────────────────
    clean = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON response from model",
            "raw_output": raw
        }


# ── Sync Wrapper (unchanged) ──────────────────────────────────────────────────
def get_ai_recommendations_sync(
    channel_summary: list[dict],
    cohort_summary: list[dict],
    total_budget: float = 500_000,
) -> dict:
    """Synchronous wrapper used by Streamlit."""
    import asyncio
    return asyncio.run(get_ai_recommendations(channel_summary, cohort_summary, total_budget))