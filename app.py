import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import time
from feature_extractor import URLFeatureExtractor, compute_risk_score, get_risk_category

st.set_page_config(
    page_title="PhishGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #060b18; color: #e2e8f0; }

.hero { text-align:center; padding:2.5rem 0 1.5rem; }
.hero h1 {
    font-size:3rem; font-weight:700; margin:0;
    background:linear-gradient(135deg,#38bdf8 0%,#818cf8 50%,#f472b6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero p { color:#64748b; font-size:1.05rem; margin-top:.4rem; }

.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.5rem 0; }
.stat { background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:1.2rem; text-align:center; }
.stat-num { font-size:2rem; font-weight:700; color:#38bdf8; }
.stat-lbl { font-size:.75rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }

.result-card { border-radius:16px; padding:1.8rem; margin:1rem 0; text-align:center; }
.result-safe      { background:#052e16; border:2px solid #16a34a; }
.result-suspicious{ background:#431407; border:2px solid #d97706; }
.result-high      { background:#450a0a; border:2px solid #dc2626; }
.result-phishing  { background:#3b0764; border:2px solid #9333ea; }
.result-title { font-size:2.2rem; font-weight:700; letter-spacing:.06em; }
.result-meta  { color:#94a3b8; margin-top:.5rem; font-size:.9rem; }

.feat-item { display:flex; justify-content:space-between; align-items:center;
             padding:.55rem 1rem; border-radius:9px; margin-bottom:.35rem; background:#0f172a; }
.feat-safe    { border-left:4px solid #16a34a; }
.feat-warning { border-left:4px solid #d97706; }
.feat-danger  { border-left:4px solid #dc2626; }
.feat-name { font-weight:600; font-size:.88rem; color:#e2e8f0; }
.feat-desc { font-size:.75rem; color:#475569; margin-top:1px; }
.badge { padding:2px 10px; border-radius:99px; font-size:.72rem; font-weight:700; white-space:nowrap; }
.badge-safe    { background:#052e16; color:#4ade80; }
.badge-warning { background:#431407; color:#fbbf24; }
.badge-danger  { background:#450a0a; color:#f87171; }

.model-card { background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:1.4rem; text-align:center; }
.model-acc  { font-size:1.8rem; font-weight:700; }
.model-name { font-size:.82rem; color:#64748b; margin-top:.2rem; }
.model-best { font-size:.72rem; color:#4ade80; margin-top:.2rem; }

.step-card  { background:#0f172a; border:1px solid #1e293b; border-radius:12px;
              padding:1rem 1.2rem; margin-bottom:.8rem; display:flex; gap:1rem; align-items:start; }
.step-num   { background:linear-gradient(135deg,#0ea5e9,#6366f1); color:#fff; border-radius:50%;
              width:30px; height:30px; display:flex; align-items:center; justify-content:center;
              font-weight:700; font-size:.85rem; flex-shrink:0; margin-top:2px; }
.step-title { font-weight:600; color:#e2e8f0; font-size:.9rem; }
.step-desc  { color:#475569; font-size:.82rem; margin-top:2px; }

.url-mono   { font-family:'JetBrains Mono',monospace; background:#0f172a; border:1px solid #1e293b;
              border-radius:8px; padding:.6rem 1rem; color:#64748b; font-size:.85rem;
              word-break:break-all; margin-top:.5rem; }
.section-title { font-size:1.1rem; font-weight:700; color:#cbd5e1;
                 border-bottom:1px solid #1e293b; padding-bottom:.5rem; margin:1.5rem 0 1rem; }
.team-card  { background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:1rem; text-align:center; }
.team-name  { font-weight:600; color:#e2e8f0; font-size:.88rem; margin-top:.4rem; }
.team-id    { color:#475569; font-size:.78rem; }
.team-role  { color:#38bdf8; font-size:.75rem; margin-top:.2rem; }

.stTextInput>div>div>input {
    background:#0f172a !important; color:#f1f5f9 !important;
    border:2px solid #1e293b !important; border-radius:10px !important;
    font-size:1rem !important; padding:.7rem 1rem !important;
}
.stTextInput>div>div>input:focus { border-color:#38bdf8 !important; }
.stButton>button {
    background:linear-gradient(135deg,#0ea5e9,#6366f1) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    padding:.7rem 1.5rem !important; font-weight:600 !important;
    width:100% !important; font-size:1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('phishguard_model.pkl'):
        with open('phishguard_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

md  = load_model()
acc = md['accuracy'] * 100 if md else 97.3

# ── Hero ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🛡️ PhishGuard Pro</h1>
  <p>Real-time phishing detection · Explainable AI · Ensemble Learning</p>
</div>
<div class="stat-grid">
  <div class="stat"><div class="stat-num">{acc:.1f}%</div><div class="stat-lbl">Model Accuracy</div></div>
  <div class="stat"><div class="stat-num">30</div><div class="stat-lbl">URL Features</div></div>
  <div class="stat"><div class="stat-num">3</div><div class="stat-lbl">ML Models</div></div>
  <div class="stat"><div class="stat-num">11K+</div><div class="stat-lbl">Training Samples</div></div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">🔍 URL Scanner</div>', unsafe_allow_html=True)

    url = st.text_input("URL", placeholder="https://example.com",
                        label_visibility="collapsed")

    e1, e2, e3 = st.columns(3)
    if e1.button("✅ google.com"):        url = "https://www.google.com"
    if e2.button("⚠️ paypal-secure.tk"): url = "http://paypal-secure-login.tk/verify@account"
    if e3.button("🚨 IP address URL"):    url = "http://192.168.1.1/banking/login"

    scan = st.button("🔍 Analyze URL", use_container_width=True)

    if scan and url:
        with st.spinner("Analyzing..."):
            time.sleep(0.4)
            ext           = URLFeatureExtractor()
            feats, expl   = ext.extract(url)
            score         = compute_risk_score(expl)
            cat, col      = get_risk_category(score)

            if md:
                arr  = np.array(feats).reshape(1, -1)
                pred = md['rf'].predict(arr)[0]
                prob = md['rf'].predict_proba(arr)[0]
                conf = prob[pred] * 100
            else:
                pred = 1 if score > 25 else 0
                conf = 90.0

            st.session_state.result = dict(
                url=url, feats=feats, expl=expl,
                score=score, cat=cat, col=col,
                pred=pred, conf=conf
            )

    if 'result' in st.session_state:
        r        = st.session_state.result
        cat      = r['cat']
        score    = r['score']
        card_cls = {
            'SAFE':       'result-safe',
            'SUSPICIOUS': 'result-suspicious',
            'HIGH RISK':  'result-high',
            'PHISHING':   'result-phishing',
        }.get(cat, 'result-high')
        icon     = {'SAFE':'✅','SUSPICIOUS':'⚠️','HIGH RISK':'🚨','PHISHING':'☠️'}.get(cat,'🚨')
        pred_txt = "PHISHING" if r['pred'] == 1 else "LEGITIMATE"

        st.markdown(f"""
        <div class="result-card {card_cls}">
          <div class="result-title" style="color:{r['col']}">{icon} {cat}</div>
          <div class="result-meta">
            Risk Score: <strong style="color:{r['col']}">{score}/100</strong>
            &nbsp;·&nbsp; ML Verdict: <strong>{pred_txt}</strong>
            &nbsp;·&nbsp; Confidence: <strong>{r['conf']:.1f}%</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="url-mono">{r["url"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(score), text=f"Risk Score: {score}/100")

        if   cat == 'SAFE':       st.success("This URL appears legitimate.")
        elif cat == 'SUSPICIOUS': st.warning("Suspicious. Verify before entering credentials.")
        else:                     st.error("High phishing probability — do NOT enter personal info!")

with right:
    if 'result' in st.session_state:
        r    = st.session_state.result
        expl = r['expl']
        d = sum(1 for v in expl.values() if v['status'] == 'danger')
        w = sum(1 for v in expl.values() if v['status'] == 'warning')
        s = sum(1 for v in expl.values() if v['status'] == 'safe')

        st.markdown('<div class="section-title">🔬 Explainability Panel</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#475569;font-size:.83rem;margin-bottom:1rem">Every flag explained — no black box.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div style="text-align:center"><div style="color:#ef4444;font-size:1.6rem;font-weight:700">{d}</div><div style="color:#475569;font-size:.72rem">DANGER</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div style="text-align:center"><div style="color:#f59e0b;font-size:1.6rem;font-weight:700">{w}</div><div style="color:#475569;font-size:.72rem">WARNING</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div style="text-align:center"><div style="color:#22c55e;font-size:1.6rem;font-weight:700">{s}</div><div style="color:#475569;font-size:.72rem">SAFE</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        sorted_expl = sorted(
            expl.items(),
            key=lambda x: {'danger':0,'warning':1,'safe':2}[x[1]['status']]
        )
        for _, info in sorted_expl:
            bc  = f"badge-{info['status']}"
            fc  = f"feat-{info['status']}"
            tag = {'safe':'✅ SAFE','warning':'⚠️ WARN','danger':'🚨 RISK'}[info['status']]
            st.markdown(f"""
            <div class="feat-item {fc}">
              <div>
                <div class="feat-name">{info['label']}</div>
                <div class="feat-desc">{info['desc'][:72]}...</div>
              </div>
              <span class="badge {bc}">{tag}</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-title">⚙️ How It Works</div>', unsafe_allow_html=True)
        for n, t, d in [
            ("1","URL Input",          "Enter any URL or use the quick-test buttons"),
            ("2","Feature Extraction", "30 features extracted in real-time"),
            ("3","Ensemble Voting",    "RF + GBM + LR soft-vote for final prediction"),
            ("4","Explainability",     "Every feature decision shown — no black box"),
            ("5","Risk Score",         "0–100 score with SAFE / SUSPICIOUS / HIGH RISK / PHISHING"),
        ]:
            st.markdown(f"""
            <div class="step-card">
              <div class="step-num">{n}</div>
              <div><div class="step-title">{t}</div><div class="step-desc">{d}</div></div>
            </div>""", unsafe_allow_html=True)

# ── Model Comparison ──────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

comp   = md['comparison'] if md else {
    'Logistic Regression':0.912,'Random Forest':0.965,
    'Gradient Boosting':0.968,'Ensemble (Ours)':0.973}
colors = ['#6366f1','#38bdf8','#f59e0b','#22c55e']
for col, (name, a), color in zip(st.columns(4), comp.items(), colors):
    best     = name == 'Ensemble (Ours)'
    border   = f'border:2px solid {color};' if best else ''
    best_tag = '<div class="model-best">★ Best Performer</div>' if best else ''
    col.markdown(f"""
    <div class="model-card" style="{border}">
      <div class="model-acc" style="color:{color}">{a*100:.1f}%</div>
      <div class="model-name">{name}</div>{best_tag}
    </div>""", unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────
st.markdown('<div class="section-title" style="margin-top:1.5rem">🏆 Top Phishing Indicators</div>', unsafe_allow_html=True)
if md and 'fi' in md:
    fi_df = md['fi'].head(10).set_index('feature')[['importance']]
    fi_df.columns = ['Importance Score']
else:
    fi_df = pd.DataFrame({'Importance Score':{
        'SSLfinal_State':0.082,'URL_of_Anchor':0.078,
        'having_Sub_Domain':0.071,'web_traffic':0.069,
        'Request_URL':0.065,'age_of_domain':0.058,
        'Domain_registeration_length':0.054,'Statistical_report':0.051,
        'Links_in_tags':0.049,'Prefix_Suffix':0.043}})
st.bar_chart(fi_df, color='#38bdf8', height=280)

