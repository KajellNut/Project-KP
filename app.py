import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

# ── KONFIGURASI ────────────────────────────────────────────────────
SHEET_URL       = st.secrets["SHEET_URL"]
JADWAL_URL      = st.secrets["JADWAL_URL"]  
N8N_WEBHOOK_URL = st.secrets["N8N_WEBHOOK_URL"]

BATAS_MW            = 90.0    # Batas minimum beban CT (MW) — spec PLN MCTN
BATAS_HRSG_FALLBACK = 7000.0  # Fallback HRSG jika data kurang

# Timezone WIB = UTC+7
WIB = timezone(timedelta(hours=7))

# EOH Milestone PLN MCTN
MILESTONE = {"CI": 15984, "HGPI": 31968, "MO": 63936}

STATUS_COLOR = {
    "AMAN":        "#22c55e",
    "WARNING":     "#f59e0b",
    "KRITIS":      "#ef4444",
    "MAINTENANCE": "#3b82f6",
    "DATA MINIM":  "#6b7280",
    "OFF / TRIP":  "#6b7280",
}

# ── HELPERS ────────────────────────────────────────────────────────
def now_wib():
    return datetime.now(WIB).replace(tzinfo=None)

def parse_tgl(s):
    if not s or str(s).strip() in ["", "-", "nan"]:
        return None
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return None

def ekstrak_eoh(keterangan):
    """Ekstrak angka EOH dari string seperti '40,760 EOH (127%)'"""
    if not keterangan or str(keterangan).strip() in ["", "-", "nan"]:
        return None
    m = re.search(r"([\d,\.]+)\s*EOH", str(keterangan), re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "").replace(".", ""))
    except:
        return None

def hitung_eoh(unit, df_jadwal):
    today = now_wib()
    uj = df_jadwal[df_jadwal["unit"] == unit].copy()
    if uj.empty:
        return None

    uj["_startup"]  = uj["tanggal_start_up"].apply(parse_tgl)
    uj["_shutdown"] = uj["tanggal_shut_down"].apply(parse_tgl)
    uj = uj.dropna(subset=["_startup"]).sort_values("_startup", ascending=False)
    last = uj.iloc[0]

    # EOH saat masuk maintenance terakhir (dari keterangan_scope)
    eoh_base = ekstrak_eoh(last.get("keterangan_scope", ""))
    if eoh_base is None:
        raw = str(last.get("target_EOH", "0")).strip().replace(",", "").replace(".", "")
        try:
            eoh_base = float(raw) if raw and raw not in ["-", ""] else 0
        except:
            eoh_base = 0

    startup_dt  = last["_startup"]
    hari_jalan  = max(0, (today - startup_dt).days)
    current_eoh = round(eoh_base + hari_jalan * 24)

    # Jadwal maintenance berikutnya (tanggal_shut_down > hari ini)
    jadwal_depan = uj[
        uj["_shutdown"].apply(lambda d: d is not None and d > today)
    ].sort_values("_shutdown")

    sisa_hari       = None
    next_maint_type = "N/A"
    next_maint_date = None
    if not jadwal_depan.empty:
        nxt             = jadwal_depan.iloc[0]
        next_maint_date = nxt["_shutdown"]
        next_maint_type = str(nxt.get("jenis_maintenance", "N/A"))
        sisa_hari       = max(0, (next_maint_date - today).days)

    # Progress bar: % menuju jadwal berikutnya berdasarkan milestone interval
    if "Major" in next_maint_type or "MO" in next_maint_type:
        ms_val = MILESTONE["MO"]
    elif "HGPI" in next_maint_type or "Hot Gas" in next_maint_type:
        ms_val = MILESTONE["HGPI"]
    else:
        ms_val = MILESTONE["CI"]

    if sisa_hari is not None and ms_val > 0:
        sisa_eoh = sisa_hari * 24
        eoh_used = max(0, ms_val - sisa_eoh)
        persen   = min(round(eoh_used / ms_val * 100), 100)
    else:
        persen = 0

    bar_color = "#ef4444" if persen >= 85 else "#f59e0b" if persen >= 65 else "#22c55e"

    return {
        "unit":             unit,
        "current_EOH":      current_eoh,
        "next_ms_type":     next_maint_type,
        "persentase":       persen,
        "bar_color":        bar_color,
        "sisa_hari":        sisa_hari,
        "next_maint_date":  next_maint_date.strftime("%d %b %Y") if next_maint_date else "N/A",
        "last_maintenance": last.get("jenis_maintenance", "N/A"),
    }

def hitung_regresi(series, batas):
    y        = pd.to_numeric(series, errors="coerce").fillna(0)
    y_active = y[y > 0].values
    if len(y_active) < 2:
        return {"slope": 0, "days_left": None, "status": "DATA MINIM", "y_trend": None, "r2": 0}

    X         = np.arange(len(y_active)).reshape(-1, 1)
    model     = LinearRegression().fit(X, y_active)
    slope     = round(float(model.coef_[0]), 4)
    intercept = float(model.intercept_)
    y_trend   = model.predict(X).tolist()

    ss_tot = np.sum((y_active - np.mean(y_active)) ** 2)
    ss_res = np.sum((y_active - model.predict(X).flatten()) ** 2)
    r2     = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 1.0

    if slope >= 0:
        return {"slope": slope, "days_left": 999, "status": "AMAN", "y_trend": y_trend, "r2": r2}

    days_left = max(0, round((batas - intercept) / slope - len(y_active)))
    status    = "KRITIS" if days_left <= 3 else "WARNING" if days_left <= 14 else "AMAN"
    return {"slope": slope, "days_left": days_left, "status": status, "y_trend": y_trend, "r2": r2}

def hitung_batas_hrsg(df, col):
    y = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    y = y[y > 0].dropna()
    if len(y) < 3:
        return BATAS_HRSG_FALLBACK
    return max(BATAS_HRSG_FALLBACK, round(y.mean() - y.std(), 0))

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PLN MCTN — Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0a0e1a; color: #e2e8f0; }
.stApp { background-color: #0a0e1a; }
.header-band {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f; border-radius: 12px; padding: 28px 36px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.header-band::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #3b82f6);
}
.header-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #e2e8f0; margin: 0; }
.header-sub   { font-size: 0.85rem; color: #64748b; margin: 6px 0 0; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; }
.ts-badge {
    display: inline-block; background: #1e3a5f; color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    padding: 4px 10px; border-radius: 20px; margin-top: 10px; border: 1px solid #2d4f7c;
}
.card {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 10px;
    padding: 20px; margin-bottom: 16px; position: relative; overflow: hidden;
}
.card-accent { position: absolute; top: 0; left: 0; width: 4px; height: 100%; border-radius: 10px 0 0 10px; }
.card-unit   { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.card-val    { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1.1; }
.card-sub    { font-size: 0.75rem; color: #64748b; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px;
}
.eoh-bar-bg   { background: #1e293b; border-radius: 4px; height: 8px; margin: 8px 0 4px; overflow: hidden; }
.eoh-bar-fill { height: 100%; border-radius: 4px; }
.section-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #475569;
    letter-spacing: 2px; text-transform: uppercase; margin: 24px 0 12px;
    padding-bottom: 6px; border-bottom: 1px solid #1e293b;
}
.ai-card { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 20px 24px; margin-bottom: 12px; }
.ai-unit-badge { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; }
.ai-advice { font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; margin-top: 8px; }
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af); color: white; border: none;
    border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    padding: 12px 28px; width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    for c in ["ct1_mw","ct2_mw","ct3_mw","hrsg1_ton","hrsg2_ton","hrsg3_ton","dm_net_mw","cadangan_mw","total_produksi_uap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_jadwal():
    return pd.read_csv(JADWAL_URL)

try:
    df        = load_data()
    df_jadwal = load_jadwal()
    data_ok   = True
except Exception as e:
    data_ok = False
    err_msg = str(e)

# ── HEADER ────────────────────────────────────────────────────────
now_str = datetime.now(WIB).strftime("%d %b %Y — %H:%M WIB")
st.markdown(
    '<div class="header-band">'
    '<p class="header-title">⚡ PLN MCTN — Predictive Maintenance System</p>'
    '<p class="header-sub">SISTEM PENDUKUNG KEPUTUSAN PEMELIHARAAN PREDIKTIF UNIT PEMBANGKIT</p>'
    f'<span class="ts-badge">🕐 {now_str}</span>'
    '</div>',
    unsafe_allow_html=True
)

if not data_ok:
    st.error(f"❌ Gagal memuat data: {err_msg}")
    st.stop()

# ── HITUNG METRIK ─────────────────────────────────────────────────
units_ct   = {"CT1": "ct1_mw",   "CT2": "ct2_mw",   "CT3": "ct3_mw"}
units_hrsg = {"HRSG1": "hrsg1_ton", "HRSG2": "hrsg2_ton", "HRSG3": "hrsg3_ton"}

ket_terakhir = ""
if "keterangan" in df.columns:
    ket_terakhir = " ".join(df["keterangan"].dropna().tail(3).astype(str).tolist()).lower()

results  = {}
eoh_data = {}

for name, col in {**units_ct, **units_hrsg}.items():
    if col not in df.columns:
        continue
    batas = BATAS_MW if "CT" in name else hitung_batas_hrsg(df, col)
    cur   = float(df[col].dropna().iloc[-1]) if not df[col].dropna().empty else 0

    if cur <= 0:
        maint_kw = ["maintenance","repair","s/d","har","inspeksi","hgpi","overhaul","shutdown"]
        status   = "MAINTENANCE" if any(w in ket_terakhir for w in maint_kw) else "OFF / TRIP"
        results[name] = {"val": cur, "status": status, "days_left": "N/A", "slope": 0, "r2": 0, "y_trend": None}
    else:
        reg = hitung_regresi(df[col], batas)
        if cur < batas:
            reg["status"] = "KRITIS"
        results[name] = {"val": cur, "status": reg["status"],
                         "days_left": reg["days_left"], "slope": reg["slope"],
                         "r2": reg["r2"], "y_trend": reg["y_trend"]}

for unit in ["CT1", "CT2", "CT3"]:
    eoh = hitung_eoh(unit, df_jadwal)
    if eoh:
        eoh_data[unit] = eoh
        sisa       = eoh.get("sisa_hari")
        cur_status = results.get(unit, {}).get("status", "")
        if sisa is not None and cur_status not in ["MAINTENANCE", "OFF / TRIP"]:
            if sisa <= 14:
                results[unit]["status"] = "KRITIS"
            elif sisa <= 30 and cur_status == "AMAN":
                results[unit]["status"] = "WARNING"

# ── SECTION 1: STATUS CT ──────────────────────────────────────────
st.markdown('<p class="section-title">// STATUS UNIT PEMBANGKIT</p>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (name, col) in enumerate(units_ct.items()):
    r      = results.get(name, {})
    eoh    = eoh_data.get(name, {})
    color  = STATUS_COLOR.get(r.get("status", ""), "#6b7280")
    val    = r.get("val", 0)
    status = r.get("status", "N/A")
    dl     = r.get("days_left", "N/A")

    val_str   = "OFF" if val <= 0 else f"{val:.2f} MW"
    dl_str    = "Stabil ↗" if dl == 999 else (f"{dl} hari" if isinstance(dl, (int, float)) else str(dl))
    slope_val = r.get("slope", 0)
    slope_str = f"&nbsp;|&nbsp;Slope: {slope_val} MW/hari" if slope_val != 0 else ""
    sub_str   = f"Estimasi: {dl_str}{slope_str}"

    # EOH bar — pure string concat, no nested f-string conditionals
    eoh_html = ""
    if eoh:
        pct       = eoh.get("persentase", 0)
        bar_color = eoh.get("bar_color", "#22c55e")
        cur_eoh   = eoh.get("current_EOH", 0)
        ms_type   = eoh.get("next_ms_type", "N/A")
        ndate     = eoh.get("next_maint_date", "N/A")
        sisa      = eoh.get("sisa_hari")
        sisa_str  = f"~{sisa} hari" if sisa is not None else "N/A"
        eoh_html = (
            '<div style="margin-top:12px;">'
            + '<div style="display:flex;justify-content:space-between;'
            + 'font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#64748b;">'
            + f'<span>EOH: {cur_eoh:,} jam</span>'
            + f'<span>{pct}% menuju {ms_type}</span>'
            + '</div>'
            + '<div class="eoh-bar-bg">'
            + f'<div class="eoh-bar-fill" style="width:{pct}%;background:{bar_color};"></div>'
            + '</div>'
            + '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#475569;">'
            + f'Jadwal: {ms_type} pada {ndate} (sisa {sisa_str})'
            + '</div>'
            + '</div>'
        )

    with cols[i]:
        st.markdown(
            '<div class="card">'
            + f'<div class="card-accent" style="background:{color};"></div>'
            + '<div style="padding-left:12px;">'
            + f'<div class="card-unit">{name}</div>'
            + f'<div class="card-val" style="color:{color};">{val_str}</div>'
            + '<div style="margin-top:6px;">'
            + f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{status}</span>'
            + '</div>'
            + f'<div class="card-sub" style="margin-top:8px;">{sub_str}</div>'
            + eoh_html
            + '</div>'
            + '</div>',
            unsafe_allow_html=True
        )

# ── SECTION 2: STATUS HRSG ────────────────────────────────────────
st.markdown('<p class="section-title">// STATUS HRSG</p>', unsafe_allow_html=True)
cols2 = st.columns(3)
for i, (name, col) in enumerate(units_hrsg.items()):
    r       = results.get(name, {})
    color   = STATUS_COLOR.get(r.get("status", ""), "#6b7280")
    val     = r.get("val", 0)
    status  = r.get("status", "N/A")
    val_str = "OFF" if val <= 0 else f"{val:,.0f} TON"

    with cols2[i]:
        st.markdown(
            '<div class="card">'
            + f'<div class="card-accent" style="background:{color};"></div>'
            + '<div style="padding-left:12px;">'
            + f'<div class="card-unit">{name}</div>'
            + f'<div class="card-val" style="color:{color};">{val_str}</div>'
            + '<div style="margin-top:6px;">'
            + f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{status}</span>'
            + '</div>'
            + '</div>'
            + '</div>',
            unsafe_allow_html=True
        )

# ── SECTION 3: GRAFIK TREN ────────────────────────────────────────
st.markdown('<p class="section-title">// ANALISIS TREN BEBAN</p>', unsafe_allow_html=True)

def buat_grafik(df, col, name, batas, color, result):
    fig    = go.Figure()
    y_all  = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    y_plot = y_all.replace(0, np.nan)
    fig.add_trace(go.Scatter(x=list(range(len(y_all))), y=y_plot, name="Aktual",
                             line=dict(color=color, width=2), connectgaps=False))
    y_trend = result.get("y_trend")
    if y_trend:
        active_idx = [j for j, v in enumerate(y_all) if v > 0]
        if len(active_idx) == len(y_trend):
            fig.add_trace(go.Scatter(x=active_idx, y=y_trend, name="Tren",
                                     line=dict(color="#f59e0b", width=2, dash="dot")))
    fig.add_hline(y=batas, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Batas {batas}", annotation_position="top right")
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
        title=dict(text=f"Tren Beban {name}", font=dict(color="#e2e8f0", size=14)),
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b", showgrid=True),
        yaxis=dict(gridcolor="#1e293b", showgrid=True),
    )
    return fig

tab1, tab2 = st.tabs(["📈 Tren CT (MW)", "💧 Tren HRSG (Ton)"])
with tab1:
    cc = st.columns(3)
    for i, (name, col) in enumerate(units_ct.items()):
        if col in df.columns:
            with cc[i]:
                st.plotly_chart(buat_grafik(df, col, name, BATAS_MW,
                    STATUS_COLOR.get(results.get(name, {}).get("status", ""), "#3b82f6"),
                    results.get(name, {})), use_container_width=True)
with tab2:
    hc = st.columns(3)
    for i, (name, col) in enumerate(units_hrsg.items()):
        if col in df.columns:
            with hc[i]:
                st.plotly_chart(buat_grafik(df, col, name, hitung_batas_hrsg(df, col),
                    STATUS_COLOR.get(results.get(name, {}).get("status", ""), "#06b6d4"),
                    results.get(name, {})), use_container_width=True)

# ── SECTION 4: AI ADVISOR ─────────────────────────────────────────
st.markdown('<p class="section-title">// AI MAINTENANCE ADVISOR</p>', unsafe_allow_html=True)

batch_results = []
for name in ["CT1", "CT2", "CT3"]:
    r   = results.get(name, {})
    eoh = eoh_data.get(name, {})
    batch_results.append({
        "unit":             name,
        "status":           r.get("status", "N/A"),
        "days_left":        r.get("days_left", "N/A"),
        "current_mw":       r.get("val", 0),
        "slope":            r.get("slope", 0),
        "r2":               r.get("r2", 0),
        "current_EOH":      eoh.get("current_EOH", "N/A"),
        "eoh_dalam_siklus": eoh.get("current_EOH", 0) % MILESTONE["MO"],
        "next_milestone":   eoh.get("next_ms_type", "N/A"),
        "sisa_hari_jadwal": eoh.get("sisa_hari", "N/A"),
        "next_maint_date":  eoh.get("next_maint_date", "N/A"),
        "persentase_EOH":   eoh.get("persentase", "N/A"),
    })

ket_payload = ""
if "keterangan" in df.columns:
    ket_payload = " | ".join(df["keterangan"].dropna().tail(3).astype(str).tolist())

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    run_ai = st.button("🤖 Jalankan Analisis AI Terintegrasi", use_container_width=True)
with col_btn2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if run_ai:
    with st.spinner("AI sedang menganalisis kondisi unit dan laporan operasional..."):
        try:
            payload = {
                "timestamp":      datetime.now(WIB).strftime("%Y-%m-%d %H:%M"),
                "plant_summary":  batch_results,
                "operator_notes": ket_payload
            }
            resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                ai_raw = resp.json()
                data   = ai_raw if isinstance(ai_raw, list) else ai_raw.get("data_tabel", [])
                st.session_state.ai_results = data
            else:
                st.error(f"n8n response: {resp.status_code}")
        except Exception as e:
            st.error(f"Koneksi ke n8n gagal: {e}")

# Placeholder permanen untuk hasil AI
ai_placeholder = st.empty()

if st.session_state.ai_results:
    with ai_placeholder.container():
        st.success("✅ Analisis AI berhasil")
        for item in st.session_state.ai_results:
            unit   = item.get("unit", "N/A")
            status = item.get("status", "N/A")
            advice = item.get("advice", "Tidak ada saran.")
            dl     = item.get("days_left", "N/A")
            color  = STATUS_COLOR.get(status, "#6b7280")
            dl_str = f"{dl} hari" if isinstance(dl, (int, float)) and str(dl) != "N/A" else str(dl)

            lines = [p.strip().strip("-").strip()
                     for p in advice.replace("\n", ". ").split(". ")
                     if len(p.strip()) > 10]

            if len(lines) > 1:
                items_html = "".join(
                    '<div style="display:flex;gap:8px;margin-bottom:6px;">'
                    + f'<span style="color:{color};flex-shrink:0;">▸</span>'
                    + f'<span>{ln}{"." if not ln.endswith(".") else ""}</span>'
                    + '</div>'
                    for ln in lines
                )
                advice_html = f'<div style="margin-top:12px;">{items_html}</div>'
            else:
                advice_html = f'<div class="ai-advice">{advice}</div>'

            st.markdown(
                '<div class="ai-card">'
                + '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px;">'
                + f'<span class="ai-unit-badge" style="color:{color};">⚡ {unit}</span>'
                + f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{status}</span>'
                + f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#475569;">Estimasi beban: {dl_str}</span>'
                + '</div>'
                + advice_html
                + '</div>',
                unsafe_allow_html=True
            )
else:
    with ai_placeholder.container():
        st.info("💡 Tekan tombol di atas untuk menjalankan analisis AI.")

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:24px 0 8px;font-family:\'IBM Plex Mono\',monospace;'
    'font-size:0.7rem;color:#334155;border-top:1px solid #1e293b;margin-top:32px;">'
    'PLN MCTN Predictive Maintenance System — Developed for KP Project'
    '</div>',
    unsafe_allow_html=True
)