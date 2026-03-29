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

# ── HELPER: PARSE TANGGAL DD/MM/YYYY atau YYYY-MM-DD ───────────────
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

# ── HELPER: EKSTRAK EOH AKTUAL DARI keterangan_scope ──────────────
def ekstrak_eoh_dari_keterangan(keterangan):
    """
    Ekstrak angka EOH aktual dari string seperti:
    "40,760 EOH (127%) Generator Rotor Replacement"
    "18,720 EOH (117%)"
    "15,984 EOH (100%)"
    Return float atau None jika tidak ditemukan.
    """
    import re
    if not keterangan or str(keterangan).strip() in ["", "-", "nan"]:
        return None
    # Cari pola angka sebelum kata EOH
    match = re.search(r"([\d,\.]+)\s*EOH", str(keterangan), re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).replace(",", "").replace(".", "")
    try:
        return float(raw)
    except:
        return None

# ── HELPER: HITUNG EOH OTOMATIS ────────────────────────────────────
def hitung_eoh(unit, df_jadwal):
    today = datetime.now(WIB).replace(tzinfo=None)
    unit_jadwal = df_jadwal[df_jadwal["unit"] == unit].copy()
    if unit_jadwal.empty:
        return None

    unit_jadwal["_startup"] = unit_jadwal["tanggal_start_up"].apply(parse_tgl)
    unit_jadwal = unit_jadwal.dropna(subset=["_startup"])
    unit_jadwal = unit_jadwal.sort_values("_startup", ascending=False)

    last = unit_jadwal.iloc[0]

    # Cari EOH aktual saat shutdown dari keterangan_scope
    # Format: "40,760 EOH (127%) Generator Rotor Replacement"
    eoh_saat_shutdown = ekstrak_eoh_dari_keterangan(last.get("keterangan_scope", ""))

    # Fallback: jika keterangan kosong, gunakan target_EOH sebagai estimasi
    if eoh_saat_shutdown is None:
        raw = str(last.get("target_EOH", "0")).strip().replace(",", "").replace(".", "")
        try:
            eoh_saat_shutdown = float(raw) if raw and raw != "-" else 0
        except:
            eoh_saat_shutdown = 0

    startup_dt = last["_startup"]
    hari_jalan = max(0, (today - startup_dt).days)

    # current_EOH = EOH saat unit kembali beroperasi (startup) + jam jalan sejak itu
    current_eoh = round(eoh_saat_shutdown + hari_jalan * 24)

    # target_EOH = milestone yang memicu maintenance ini (CI/HGPI/MO)
    # Gunakan ini untuk tahu kita sudah di siklus keberapa
    raw_target = str(last.get("target_EOH", "0")).strip().replace(",", "").replace(".", "")
    try:
        target_eoh_milestone = float(raw_target) if raw_target and raw_target != "-" else 0
    except:
        target_eoh_milestone = 0

    # Cari milestone BERIKUTNYA dari current_EOH
    # Milestone berlaku berulang: CI → HGPI → MO → CI → HGPI → MO → ...
    # Hitung berapa siklus MO sudah terlewati
    siklus_mo = int(current_eoh // MILESTONE["MO"])
    base       = siklus_mo * MILESTONE["MO"]

    next_type   = None
    next_target = None
    for k, v in MILESTONE.items():
        kandidat = base + v
        if kandidat > current_eoh:
            next_type   = k
            next_target = kandidat
            break

    if next_type is None:
        # Lewati semua dalam siklus ini, ke MO berikutnya
        next_type   = "MO"
        next_target = (siklus_mo + 1) * MILESTONE["MO"]

    sisa_eoh  = max(0, next_target - current_eoh)
    sisa_hari = round(sisa_eoh / 24)
    persen    = min(round((current_eoh % MILESTONE["MO"] if next_type != "MO" else current_eoh % MILESTONE["MO"]) 
                          / (next_target - base) * 100), 100) if next_target > base else 100

    # Hitung persen lebih sederhana: progress menuju next_target dari base siklus
    progress_start = base + (MILESTONE.get(
        {"CI": None, "HGPI": "CI", "MO": "HGPI"}.get(next_type, "MO") or "CI",
        0
    ) if next_type != "CI" else 0)

    # Persen = seberapa jauh current_eoh dari milestone sebelumnya ke milestone berikutnya
    prev_milestone = {
        "CI":   base,
        "HGPI": base + MILESTONE["CI"],
        "MO":   base + MILESTONE["HGPI"]
    }.get(next_type, base)
    range_milestone = next_target - prev_milestone
    persen = min(round((current_eoh - prev_milestone) / range_milestone * 100), 100) if range_milestone > 0 else 100

    return {
        "unit": unit,
        "current_EOH": current_eoh,
        "eoh_saat_shutdown": eoh_saat_shutdown,
        "next_milestone_type": next_type,
        "next_milestone_target": next_target,
        "sisa_EOH": sisa_eoh,
        "sisa_hari": sisa_hari,
        "persentase": persen,
        "last_maintenance": last.get("jenis_maintenance", "N/A"),
        "last_date": str(last.get("tanggal_shut_down", "N/A"))
    }

# ── HELPER: REGRESI LINEAR ─────────────────────────────────────────
def hitung_regresi(series, batas):
    y = pd.to_numeric(series, errors="coerce").fillna(0)
    y_active = y[y > 0].values
    if len(y_active) < 2:
        return {"slope": 0, "days_left": None, "status": "DATA MINIM", "y_trend": None, "r2": 0}

    X = np.arange(len(y_active)).reshape(-1, 1)
    model = LinearRegression().fit(X, y_active)
    slope     = round(float(model.coef_[0]), 4)
    intercept = float(model.intercept_)
    y_trend   = model.predict(X).tolist()

    ss_tot = np.sum((y_active - np.mean(y_active)) ** 2)
    ss_res = np.sum((y_active - model.predict(X).flatten()) ** 2)
    r2     = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 1.0

    if slope >= 0:
        return {"slope": slope, "days_left": 999, "status": "AMAN", "y_trend": y_trend, "r2": r2}

    days_left = max(0, round((batas - intercept) / slope - len(y_active)))
    if days_left <= 3:
        status = "KRITIS"
    elif days_left <= 14:
        status = "WARNING"
    else:
        status = "AMAN"

    return {"slope": slope, "days_left": days_left, "status": status, "y_trend": y_trend, "r2": r2}

# ── HELPER: BATAS HRSG DINAMIS ────────────────────────────────────
def hitung_batas_hrsg(df, col):
    y = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    y_active = y[y > 0].dropna()
    if len(y_active) < 3:
        return BATAS_HRSG_FALLBACK
    batas = round(y_active.mean() - y_active.std(), 0)
    return max(BATAS_HRSG_FALLBACK, batas)

# ── STATUS COLOR ───────────────────────────────────────────────────
STATUS_COLOR = {
    "AMAN":        "#22c55e",
    "WARNING":     "#f59e0b",
    "KRITIS":      "#ef4444",
    "MAINTENANCE": "#3b82f6",
    "DATA MINIM":  "#6b7280",
    "OFF / TRIP":  "#6b7280",
}

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PLN MCTN — Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.stApp { background-color: #0a0e1a; }

.header-band {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-band::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #3b82f6);
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin: 6px 0 0;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}
.ts-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 20px;
    margin-top: 10px;
    border: 1px solid #2d4f7c;
}

.card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card-accent {
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    border-radius: 10px 0 0 10px;
}
.card-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.card-val {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.1;
}
.card-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}

.eoh-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 8px;
    margin: 8px 0 4px;
    overflow: hidden;
}
.eoh-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #475569;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

.ai-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.ai-unit-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.ai-advice {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
    margin-top: 8px;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    padding: 12px 28px;
    width: 100%;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    num_cols = ["ct1_mw","ct2_mw","ct3_mw","hrsg1_ton","hrsg2_ton","hrsg3_ton","dm_net_mw","cadangan_mw","total_produksi_uap"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_jadwal():
    return pd.read_csv(JADWAL_URL)

try:
    df      = load_data()
    df_jadwal = load_jadwal()
    data_ok = True
except Exception as e:
    data_ok = False
    err_msg = str(e)

# ── HEADER ────────────────────────────────────────────────────────
now_str = datetime.now(WIB).strftime("%d %b %Y — %H:%M WIB")
st.markdown(f"""
<div class="header-band">
    <p class="header-title">⚡ PLN MCTN — Predictive Maintenance System</p>
    <p class="header-sub">SISTEM PENDUKUNG KEPUTUSAN PEMELIHARAAN PREDIKTIF UNIT PEMBANGKIT</p>
    <span class="ts-badge">🕐 {now_str}</span>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error(f"❌ Gagal memuat data dari Google Sheets: {err_msg}")
    st.stop()

# ── HITUNG SEMUA METRIK ────────────────────────────────────────────
units_ct   = {"CT1": "ct1_mw",   "CT2": "ct2_mw",   "CT3": "ct3_mw"}
units_hrsg = {"HRSG1": "hrsg1_ton", "HRSG2": "hrsg2_ton", "HRSG3": "hrsg3_ton"}

# Ambil keterangan terakhir untuk deteksi maintenance
ket_terakhir = ""
if "keterangan" in df.columns:
    ket_terakhir = " ".join(df["keterangan"].dropna().tail(3).tolist()).lower()

results    = {}
eoh_data   = {}

for name, col in {**units_ct, **units_hrsg}.items():
    if col not in df.columns:
        continue
    if "CT" in name:
        batas = BATAS_MW
    else:
        batas = hitung_batas_hrsg(df, col)
    cur   = float(df[col].dropna().iloc[-1]) if not df[col].dropna().empty else 0

    if cur <= 0:
        maint_words = ["maintenance","repair","s/d","har","inspeksi","hgpi","overhaul","shutdown"]
        status = "MAINTENANCE" if any(w in ket_terakhir for w in maint_words) else "OFF / TRIP"
        results[name] = {"val": cur, "status": status, "days_left": "N/A", "slope": 0, "r2": 0, "y_trend": None}
    else:
        reg = hitung_regresi(df[col], batas)
        results[name] = {"val": cur, "status": reg["status"],
                         "days_left": reg["days_left"], "slope": reg["slope"],
                         "r2": reg["r2"], "y_trend": reg["y_trend"]}

# EOH hanya untuk CT
for unit in ["CT1","CT2","CT3"]:
    eoh = hitung_eoh(unit, df_jadwal)
    if eoh:
        eoh_data[unit] = eoh

# ── SECTION 1: STATUS UNIT ────────────────────────────────────────
st.markdown('<p class="section-title">// STATUS UNIT PEMBANGKIT</p>', unsafe_allow_html=True)

cols = st.columns(3)
for i, (name, col) in enumerate(units_ct.items()):
    r     = results.get(name, {})
    color = STATUS_COLOR.get(r.get("status",""), "#6b7280")
    val   = r.get("val", 0)
    status= r.get("status","N/A")
    dl    = r.get("days_left","N/A")
    eoh   = eoh_data.get(name, {})

    dl_str = f"{dl} hari" if isinstance(dl, (int,float)) and dl < 999 else ("Stabil ↗" if dl == 999 else str(dl))
    slope_str = f"Slope: {r.get('slope',0)} MW/hari" if r.get('slope',0) != 0 else ""
    card_sub_text = f"Estimasi: {dl_str}" + (f"&nbsp;|&nbsp;{slope_str}" if slope_str else "")

    eoh_bar = ""
    if eoh:
        pct = eoh.get("persentase", 0)
        bar_color = "#ef4444" if pct >= 90 else "#f59e0b" if pct >= 75 else "#22c55e"
        eoh_bar = f"""
        <div style="margin-top:12px;">
            <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#64748b;">
                <span>EOH: {eoh.get('current_EOH','N/A'):,} jam</span>
                <span>{pct}% → {eoh.get('next_milestone_type','N/A')}</span>
            </div>
            <div class="eoh-bar-bg">
                <div class="eoh-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#475569;">
                Sisa ~{eoh.get('sisa_hari','N/A')} hari ke {eoh.get('next_milestone_type','N/A')}
            </div>
        </div>
        """

    val_str = "OFF" if val <= 0 else f"{val:.2f} MW"

    with cols[i]:
        st.markdown(f"""
        <div class="card">
            <div class="card-accent" style="background:{color};"></div>
            <div style="padding-left:12px;">
                <div class="card-unit">{name}</div>
                <div class="card-val" style="color:{color};">{val_str}</div>
                <div style="margin-top:6px;">
                    <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{status}</span>
                </div>
                <div class="card-sub" style="margin-top:8px;">{card_sub_text}</div>
                {eoh_bar}
            </div>
        </div>
        """, unsafe_allow_html=True)

# HRSG row
st.markdown('<p class="section-title">// STATUS HRSG</p>', unsafe_allow_html=True)
cols2 = st.columns(3)
for i, (name, col) in enumerate(units_hrsg.items()):
    r     = results.get(name, {})
    color = STATUS_COLOR.get(r.get("status",""), "#6b7280")
    val   = r.get("val", 0)
    status= r.get("status","N/A")

    with cols2[i]:
        st.markdown(f"""
        <div class="card">
            <div class="card-accent" style="background:{color};"></div>
            <div style="padding-left:12px;">
                <div class="card-unit">{name}</div>
                <div class="card-val" style="color:{color};">
                    {"OFF" if val <= 0 else f"{val:,.0f} TON"}
                </div>
                <div style="margin-top:6px;">
                    <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">
                        {status}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── SECTION 2: GRAFIK TREN ────────────────────────────────────────
st.markdown('<p class="section-title">// ANALISIS TREN BEBAN</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 Tren CT (MW)", "💧 Tren HRSG (Ton)"])

def buat_grafik(df, col, name, batas, color, result):
    fig = go.Figure()
    y_all = pd.to_numeric(df[col].astype(str).str.replace(",","."), errors="coerce")
    x_all = list(range(len(y_all)))

    # Garis aktual (0 jadi NaN agar tidak menukik)
    y_plot = y_all.replace(0, np.nan)
    fig.add_trace(go.Scatter(
        x=x_all, y=y_plot,
        name="Aktual",
        line=dict(color=color, width=2),
        connectgaps=False
    ))

    # Garis tren
    y_trend = result.get("y_trend")
    if y_trend:
        active_idx = [i for i, v in enumerate(y_all) if v > 0]
        if len(active_idx) == len(y_trend):
            fig.add_trace(go.Scatter(
                x=active_idx, y=y_trend,
                name="Tren",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ))

    # Garis batas
    fig.add_hline(y=batas, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Batas {batas}", annotation_position="top right")

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
        title=dict(text=f"Tren Beban {name}", font=dict(color="#e2e8f0", size=14)),
        height=280, margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b", showgrid=True),
        yaxis=dict(gridcolor="#1e293b", showgrid=True),
    )
    return fig

with tab1:
    ct_cols = st.columns(3)
    for i, (name, col) in enumerate(units_ct.items()):
        if col in df.columns:
            color = STATUS_COLOR.get(results.get(name,{}).get("status",""), "#3b82f6")
            fig = buat_grafik(df, col, name, BATAS_MW, color, results.get(name,{}))
            with ct_cols[i]:
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    hrsg_cols = st.columns(3)
    for i, (name, col) in enumerate(units_hrsg.items()):
        if col in df.columns:
            color = STATUS_COLOR.get(results.get(name,{}).get("status",""), "#06b6d4")
            batas_hrsg = hitung_batas_hrsg(df, col)
            fig = buat_grafik(df, col, name, batas_hrsg, color, results.get(name,{}))
            with hrsg_cols[i]:
                st.plotly_chart(fig, use_container_width=True)

# ── SECTION 3: AI ADVISOR ─────────────────────────────────────────
st.markdown('<p class="section-title">// AI MAINTENANCE ADVISOR</p>', unsafe_allow_html=True)

# Siapkan payload untuk n8n
batch_results = []
for name in ["CT1","CT2","CT3"]:
    r   = results.get(name, {})
    eoh = eoh_data.get(name, {})
    batch_results.append({
        "unit":        name,
        "status":      r.get("status","N/A"),
        "days_left":   r.get("days_left","N/A"),
        "current_mw":  r.get("val", 0),
        "slope":       r.get("slope", 0),
        "r2":          r.get("r2", 0),
        "current_EOH": eoh.get("current_EOH","N/A"),
        "persentase_EOH": eoh.get("persentase","N/A"),
        "next_milestone": eoh.get("next_milestone_type","N/A"),
        "sisa_hari_EOH":  eoh.get("sisa_hari","N/A"),
    })

ket_payload = ""
if "keterangan" in df.columns:
    ket_payload = " | ".join(df["keterangan"].dropna().tail(3).astype(str).tolist())

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

col_btn1, col_btn2 = st.columns([3,1])
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
                "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
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

if st.session_state.ai_results:
    st.success("✅ Analisis AI berhasil")
    for item in st.session_state.ai_results:
        unit   = item.get("unit","N/A")
        status = item.get("status","N/A")
        advice = item.get("advice","Tidak ada saran.")
        dl     = item.get("days_left","N/A")
        color  = STATUS_COLOR.get(status, "#6b7280")
        dl_str = f"{dl} hari" if isinstance(dl,(int,float)) and str(dl) != "N/A" else str(dl)

        # Pecah advice jadi poin-poin jika mengandung tanda titik atau newline
        advice_lines = []
        for part in advice.replace("\n", ". ").split(". "):
            part = part.strip().strip("-").strip()
            if len(part) > 10:
                advice_lines.append(part)

        # Render poin-poin advice
        advice_html = ""
        if len(advice_lines) > 1:
            items_html = "".join([
                f'<div style="display:flex;gap:8px;margin-bottom:6px;">'
                f'<span style="color:{color};margin-top:2px;flex-shrink:0;">▸</span>'
                f'<span>{line}{"." if not line.endswith(".") else ""}</span>'
                f'</div>'
                for line in advice_lines
            ])
            advice_html = f'<div style="margin-top:12px;">{items_html}</div>'
        else:
            advice_html = f'<div class="ai-advice">{advice}</div>'

        st.markdown(f"""
        <div class="ai-card">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:4px;">
                <span class="ai-unit-badge" style="color:{color};">⚡ {unit}</span>
                <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{status}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#475569;">
                    Estimasi beban: {dl_str}
                </span>
            </div>
            {advice_html}
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;font-family:'IBM Plex Mono',monospace;
            font-size:0.7rem;color:#334155;border-top:1px solid #1e293b;margin-top:32px;">
    PLN MCTN Predictive Maintenance System — Developed for KP Project
</div>
""", unsafe_allow_html=True)
