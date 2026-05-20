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

Output format — always respond in valid JSON matching the exact schema provided.
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
    """Calls Gemini 2.5 Flash with strict JSON Schema constraints."""

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .streamlit/secrets.toml")

    # Configure Gemini SDK
    genai.configure(api_key=api_key)

    # Build prompt
    prompt = build_prompt(channel_summary, cohort_summary, total_budget)

    # Define strict JSON schema configuration to prevent parsing/unescaped string errors
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "executive_summary": {"type": "STRING"},
            "critical_flags": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "channel_recommendations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "channel": {"type": "STRING"},
                        "action": {"type": "STRING"},
                        "budget_shift_pct": {"type": "NUMBER"},
                        "rationale": {"type": "STRING"},
                        "expected_payback_compression_months": {"type": "NUMBER"},
                        "ltv_expansion_tactic": {"type": "STRING"}
                    },
                    "required": [
                        "channel", 
                        "action", 
                        "budget_shift_pct", 
                        "rationale", 
                        "expected_payback_compression_months", 
                        "ltv_expansion_tactic"
                    ]
                }
            },
            "ltv_expansion_plays": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "30_day_quick_wins": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "90_day_strategic_moves": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "projected_blended_roi_lift_pct": {"type": "NUMBER"}
        },
        "required": [
            "executive_summary", 
            "critical_flags", 
            "channel_recommendations", 
            "ltv_expansion_plays", 
            "30_day_quick_wins", 
            "90_day_strategic_moves", 
            "projected_blended_roi_lift_pct"
        ]
    }

    # Initialize model with system instruction
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    # Call Gemini async enforcing the schema output
    response = await model.generate_content_async(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 2000,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    # Extract and clean response text safely
    raw = response.text
    # Try to extract a JSON object from the model output. The SDK should
    # return JSON, but sometimes the model may include surrounding text or
    # code fences; handle that gracefully.
    try:
        # First try direct JSON load
        return json.loads(raw)
    except Exception:
        pass

    # Remove common code-fence wrappers and assistant tags
    cleaned = re.sub(r"(^.*?```json\s*|\s*```.*$)", "", raw, flags=re.S)
    cleaned = re.sub(r"(^.*?```\s*|\s*```.*$)", "", cleaned, flags=re.S)
    cleaned = re.sub(r"^\s*Assistant:\s*", "", cleaned, flags=re.I)
    cleaned = cleaned.strip()

    # Try to pull the first {...} JSON object found
    m = re.search(r"(\{.*\})", cleaned, flags=re.S)
    if m:
        candidate = m.group(1)
    else:
        candidate = cleaned

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        # Provide a helpful error including a snippet of the raw response
        snippet = raw[:1000] + ("..." if len(raw) > 1000 else "")
        raise ValueError(f"Failed to parse model JSON output: {e}\nResponse snippet: {snippet}") from e

    return data


def get_ai_recommendations_sync(channel_summary: list, cohort_summary: list, total_budget: float = 500_000) -> dict:
    """Synchronous wrapper for `get_ai_recommendations` for environments
    (like Streamlit) that expect blocking calls.
    It will run the async function either via `asyncio.run` or inside a
    background thread if an event loop is already running.
    """
    try:
        return asyncio.run(get_ai_recommendations(channel_summary, cohort_summary, total_budget))
    except RuntimeError:
        # Event loop is already running (common in some app frameworks).
        # Run the coroutine in a background thread.
        result_container = {}

        def _runner():
            try:
                res = asyncio.new_event_loop()
                asyncio.set_event_loop(res)
                result_container['res'] = res.run_until_complete(
                    get_ai_recommendations(channel_summary, cohort_summary, total_budget)
                )
            finally:
                try:
                    asyncio.get_event_loop().close()
                except Exception:
                    pass

        import threading
        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        return result_container.get('res')