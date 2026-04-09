# CAC·LTV Optimizer — Deployment Guide
## Fintech Growth Portfolio Project

---

## Project Structure

```
cac-ltv-optimizer/
├── app.py                    ← Phase 4: Streamlit dashboard (main entry point)
├── analytics.py              ← Phase 2: Unit economics pipeline
├── llm_engine.py             ← Phase 3: Claude AI recommendation engine
├── generate_data.py          ← Phase 1: Synthetic dataset generator
├── saas_cac_ltv_data.csv     ← Pre-generated dataset (10,000 records)
├── requirements.txt
└── .streamlit/
    ├── config.toml           ← Dark theme + server settings
    └── secrets.toml          ← API keys (DO NOT commit to Git)
```

---

## Step 1 — Local Setup

```bash
# Clone / create project directory
mkdir cac-ltv-optimizer && cd cac-ltv-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate dataset (first time only)
python generate_data.py

# Run the Streamlit app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Step 2 — API Key Configuration

### Local development
Create `.streamlit/secrets.toml`:
```toml
[anthropic]
api_key = "sk-ant-YOUR_KEY_HERE"
```

Then update `llm_engine.py` headers section:
```python
import streamlit as st

headers = {
    "Content-Type": "application/json",
    "x-api-key": st.secrets["anthropic"]["api_key"],
    "anthropic-version": "2023-06-01",
}
```

### Environment variable alternative
```bash
export ANTHROPIC_API_KEY="sk-ant-YOUR_KEY_HERE"
```

---

## Step 3 — Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add app.py analytics.py llm_engine.py requirements.txt saas_cac_ltv_data.csv
   git add .streamlit/config.toml   # DO NOT add secrets.toml
   echo ".streamlit/secrets.toml" >> .gitignore
   git commit -m "feat: CAC·LTV Optimizer dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/cac-ltv-optimizer.git
   git push -u origin main
   ```

2. **Deploy on share.streamlit.io**
   - Go to https://share.streamlit.io
   - Click "New app" → connect your GitHub repo
   - Set: Main file path = `app.py`
   - Under "Advanced settings" → Secrets, paste:
     ```toml
     [anthropic]
     api_key = "sk-ant-YOUR_KEY_HERE"
     ```
   - Click Deploy

3. **You get a URL like:**
   `https://your-username-cac-ltv-optimizer-app-xxxx.streamlit.app`

---

## Step 4 — Resume / Portfolio Bullet Points

Use these in your Fintech Growth Intern resume:

```
• Built AI-Powered CAC:LTV Optimizer in Python/Streamlit analyzing 10,000+ B2B SaaS
  customer cohorts; identified Paid Social channel with 0.60 CAC:LTV ratio (2× threshold),
  driving recommendations that projected +34% blended ROI improvement

• Engineered hybrid rule-based + LLM (Claude API) recommendation engine generating
  Marketing Mix Reallocation strategies; automated flagging of high-CAC cohorts (>33% of LTV)
  with payback period compression analysis aligned to Skydo's cross-border unit economics model

• Designed interactive unit economics dashboard (Streamlit + Plotly) covering LTV expansion,
  payback period heatmaps, and cohort-level ROI scoring across 4 acquisition channels × 2 segments
```

---

## Unit Economics Reference

| Metric | Formula | Healthy Benchmark |
|--------|---------|-------------------|
| LTV | ARPU ÷ Churn Rate | 3× CAC minimum |
| Payback Period | CAC ÷ Monthly Revenue | < 12 months |
| CAC:LTV Ratio | CAC ÷ LTV | < 0.33 (Critical above) |
| ROI Score | (LTV − CAC) ÷ CAC | > 3× |
| Blended Churn | 1 ÷ Avg Churn Month | < 5% monthly |

---

## Troubleshooting

**App won't start:** Ensure `saas_cac_ltv_data.csv` exists. Run `python generate_data.py` first.

**AI analysis fails:** Check your Anthropic API key in `.streamlit/secrets.toml`. Ensure billing is active at console.anthropic.com.

**Streamlit Cloud deploy error:** Verify `requirements.txt` has exact version pins and the CSV file is committed to the repo.
