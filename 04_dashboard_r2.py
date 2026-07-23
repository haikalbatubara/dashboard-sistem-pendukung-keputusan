"""
===============================================================================
  04 - DASHBOARD  (BAB 3.3.7)
===============================================================================
  Penulis : Muhammad Haikal Batubara
  Stage   : Dashboard Development (Streamlit)

  Mengacu Gambar 16 proposal. Halaman:
    1. Ringkasan       - KPI (total, jumlah transaksi, MAPE, anomali)
                         + grafik tren harian Aktual vs Prediksi
                         + distribusi layanan + Top biller
    2. Tren & Forecast - prediksi LSTM per layanan
    3. Drilldown       - filter per tanggal/layanan/channel/biller
    4. Anomali         - deteksi anomali via |aktual - prediksi| / sigma
    5. Evaluasi Model  - metrik global + per layanan + scatter plot

  Cara menjalankan:
      streamlit run 04_dashboard.py
===============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
DATA_DIR  = "prepared_data"
MODEL_DIR = "model_output"
EVAL_DIR  = "evaluation_output"

st.set_page_config(
    page_title="SPK M-Banking — Bank Sumut",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS KUSTOM — tampilan lebih modern & interaktif
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
}
[data-testid="stSidebar"] * { color: #e0e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.93rem; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1e2d3d;
    border: 1px solid #2e4a6a;
    border-radius: 12px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #9cb3cc !important; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #e8f4fd !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* ── Main background ── */
.main .block-container { background: #0f1923; padding-top: 1.2rem; }
.stApp { background-color: #0f1923; }

/* ── Teks umum ── */
h1, h2, h3, h4, h5, p, label, .stMarkdown { color: #d8e8f5 !important; }

/* ── Section header styling ── */
.section-header {
    border-left: 4px solid #2196F3;
    padding-left: 12px;
    margin: 1.2rem 0 0.6rem 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e8f4fd !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; }

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-green  { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
.badge-blue   { background:#1a2a3a; color:#60a5fa; border:1px solid #60a5fa; }
.badge-orange { background:#3a2a10; color:#fb923c; border:1px solid #fb923c; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOADERS  (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset_final():
    path = os.path.join(DATA_DIR, "dataset_final.csv")
    df = pd.read_csv(path, parse_dates=["received_time"])
    df["date"] = df["received_time"].dt.normalize()
    return df

@st.cache_data
def load_daily_agg():
    return pd.read_csv(os.path.join(DATA_DIR, "daily_aggregated.csv"),
                       parse_dates=["date"])

@st.cache_data
def load_predictions():
    path = os.path.join(EVAL_DIR, "predictions.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data
def load_metrics():
    path = os.path.join(EVAL_DIR, "evaluation_results.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_per_layanan_metrics():
    path = os.path.join(EVAL_DIR, "metrics_per_layanan.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# HELPER: format angka standar Indonesia (titik sebagai pemisah ribuan)
# ---------------------------------------------------------------------------
def fmt_idr(val, unit="jt", decimals=0):
    """Format rupiah dengan titik sebagai pemisah ribuan (standar Indonesia)."""
    if unit == "jt":
        num = val / 1e6
    elif unit == "m":
        num = val / 1e9
    else:
        num = val
    # format dengan koma lalu ganti koma -> titik, titik -> koma (standar ID)
    formatted = f"{num:,.{decimals}f}"
    # Python default: 1,234,567.89 -> kita ubah ke 1.234.567,89
    parts = formatted.split(".")
    integer_part = parts[0].replace(",", ".")
    if len(parts) > 1 and decimals > 0:
        return f"Rp {integer_part},{parts[1]}"
    return f"Rp {integer_part}"

def fmt_int_id(val):
    """Format integer dengan titik sebagai pemisah ribuan (Indonesia)."""
    return f"{int(val):,}".replace(",", ".")

def fmt_rp_id(val):
    """Format Rupiah lengkap tanpa satuan, titik sebagai pemisah ribuan."""
    return f"Rp {int(val):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# SIDEBAR  (navigasi & filter)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo / header
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:2.4rem;">🏦</div>
        <div style="font-weight:700; font-size:1.05rem; color:#e8f4fd;">Bank Sumut</div>
        <div style="font-size:0.78rem; color:#9cb3cc; letter-spacing:1px;">SPK M-BANKING</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigasi dengan ikon
    NAV_ICONS = {
        "Ringkasan":     "📋",
        "Tren & Forecast": "📈",
        "Drilldown":     "🔍",
        "Anomali":       "⚠️",
        "Evaluasi Model":"✏️",
    }
    page_labels = list(NAV_ICONS.keys())
    page = st.radio(
        "Navigasi",
        page_labels,
        format_func=lambda x: f"{NAV_ICONS[x]}  {x}",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem; color:#9cb3cc; letter-spacing:1px; font-weight:600;">⚙️ FILTER DATA</div>', unsafe_allow_html=True)

    df_all = load_dataset_final()

    date_min = df_all["date"].min().date()
    date_max = df_all["date"].max().date()
    st.markdown("📅 **Rentang Tanggal**")
    col_d1, col_d2 = st.columns(2)
    d_from = col_d1.date_input("Dari",   value=date_min,
                               min_value=date_min, max_value=date_max, key="d_from")
    d_to   = col_d2.date_input("Sampai", value=date_max,
                               min_value=date_min, max_value=date_max, key="d_to")

    all_lay = sorted(df_all["layanan"].unique())
    sel_lay = st.multiselect("🔗 Layanan", all_lay, default=all_lay)

    all_ch = sorted(df_all["channel_type"].dropna().unique())
    sel_ch = st.multiselect("📡 Channel", all_ch, default=all_ch)

    all_biller = sorted(df_all["biller_name"].dropna().unique())
    sel_biller = st.multiselect("🏪 Biller", all_biller, default=all_biller)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#6b8fa8; line-height:1.5;">
        <b>Tentang</b><br>
        Dashboard prediksi transaksi M-Banking berbasis model LSTM.
        Dikembangkan untuk kebutuhan operasional Bank Sumut.
    </div>
    """, unsafe_allow_html=True)

# Terapkan filter
d0, d1 = pd.to_datetime(d_from), pd.to_datetime(d_to)
if d0 > d1:
    d0, d1 = d1, d0
df_filt = df_all[(df_all["date"] >= d0) & (df_all["date"] <= d1)]
df_filt = df_filt[df_filt["layanan"].isin(sel_lay)]
df_filt = df_filt[df_filt["channel_type"].isin(sel_ch)]
df_filt = df_filt[df_filt["biller_name"].isin(sel_biller)]


# ---------------------------------------------------------------------------
# PLOTLY THEME HELPER
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,25,35,0.6)",
    font=dict(color="#c8dced", family="sans-serif", size=12),
    xaxis=dict(gridcolor="#1e3048", zeroline=False),
    yaxis=dict(gridcolor="#1e3048", zeroline=False),
    legend=dict(bgcolor="rgba(20,35,50,0.8)", bordercolor="#2e4a6a", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=50),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ===========================================================================
# HALAMAN 1 : RINGKASAN
# ===========================================================================
if page == "Ringkasan":

    # Header judul halaman
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h1 style="color:#e8f4fd; font-size:2rem; font-weight:700; margin:0 0 6px 0;">
            📋 Monitoring Transaksi
        </h1>
        <p style="color:#9cb3cc; font-size:0.9rem; margin:0;">
            Periode data: <b style="color:#4fc3f7;">01 Jan 2024 – 31 Des 2024</b>
            &nbsp;·&nbsp; Sistem Pendukung Keputusan M-Banking Bank Sumut
        </p>
        <div style="height:3px; background:linear-gradient(90deg,#2196F3,#4fc3f7,rgba(0,0,0,0));
                    border-radius:2px; margin-top:10px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ──
    metrics = load_metrics()
    preds   = load_predictions()

    total_value = df_filt["trx_value"].sum()
    total_count = len(df_filt)

    if preds is not None:
        resid        = preds["actual"] - preds["predicted"]
        sigma_resid  = resid.std() + 1e-9
        anomaly_n    = int((resid.abs() / sigma_resid > 3).sum())   # threshold >3
    else:
        anomaly_n = 0
    mape_val = metrics["overall"]["MAPE"] if metrics else None

    # Format: Total Nilai Transaksi dalam miliar Rp, pakai titik standar ID
    total_m_str = fmt_idr(total_value, unit="m", decimals=2)   # "Rp X.XXX,XX"
    # Jumlah transaksi — titik sebagai pemisah ribuan
    count_str   = fmt_int_id(total_count)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💰 Total Nilai Transaksi", total_m_str, "miliar Rp")
    with c2:
        st.metric("📊 Jumlah Transaksi", count_str, "periode terpilih")
    with c3:
        st.metric("⚠️ Anomali Terdeteksi", fmt_int_id(anomaly_n), "skor > 3σ")
    with c4:
        mape_str = f"{mape_val:.2f}%" if mape_val is not None else "-"
        st.metric("🎯 MAPE Model", mape_str, "Long Short-Term Memory")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tren Harian (full width) ──
    st.markdown('<div class="section-header">📈 Tren Harian — Aktual vs Prediksi</div>', unsafe_allow_html=True)

    if preds is not None and len(preds) > 0:
        daily = (preds.groupby("date")[["actual", "predicted"]]
                      .sum().reset_index())
        fig_tren = go.Figure()
        fig_tren.add_trace(go.Scatter(
            x=daily["date"], y=daily["actual"] / 1e6,
            name="Aktual", mode="lines+markers",
            line=dict(color="#4fc3f7", width=2),
            marker=dict(size=5),
        ))
        fig_tren.add_trace(go.Scatter(
            x=daily["date"], y=daily["predicted"] / 1e6,
            name="Prediksi (LSTM)", mode="lines+markers",
            line=dict(color="#ffb74d", width=2, dash="dot"),
            marker=dict(size=5, symbol="x"),
        ))
        fig_tren.update_layout(
            yaxis_title="Nominal Transaksi (juta Rp)",
            xaxis_title="Tanggal",
            hovermode="x unified",
            height=380,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
            xaxis=dict(gridcolor="#1e3048", zeroline=False),
            yaxis=dict(gridcolor="#1e3048", zeroline=False,
                       tickformat=",.0f"),
        )
        st.plotly_chart(fig_tren, use_container_width=True)
    else:
        daily = df_filt.groupby("date")["trx_value"].sum().reset_index()
        fig_tren = px.line(daily, x="date", y="trx_value",
                           labels={"trx_value": "Nominal Transaksi (juta Rp)", "date": "Tanggal"})
        apply_theme(fig_tren)
        st.plotly_chart(fig_tren, use_container_width=True)

    # ── Distribusi Layanan + Top Biller (side by side) ──
    col_lay, col_bill = st.columns(2)

    with col_lay:
        st.markdown('<div class="section-header">🌐 Distribusi Layanan</div>', unsafe_allow_html=True)
        lay_dist = df_filt.groupby("layanan")["trx_value"].sum().reset_index()
        lay_dist.columns = ["layanan", "total"]
        if len(lay_dist) > 0:
            COLORS = ["#4fc3f7","#81c784","#ffb74d","#e57373","#ba68c8","#4db6ac"]
            lay_dist = lay_dist.sort_values("total", ascending=False).reset_index(drop=True)

            # Buat label legend: "Nama (XX.X%)" agar info lengkap ada di legend
            total_sum = lay_dist["total"].sum()
            legend_labels = [
                f"{row['layanan']} ({row['total']/total_sum*100:.1f}%)"
                for _, row in lay_dist.iterrows()
            ]

            fig_pie = go.Figure(go.Pie(
                labels=legend_labels,          # legend pakai label lengkap
                values=lay_dist["total"],
                hole=0.45,
                name="",
                marker=dict(
                    colors=COLORS[:len(lay_dist)],
                    line=dict(color="#0f1923", width=2),
                ),
                # Hanya tampilkan % di dalam slice — tidak ada nama layanan di slice
                textinfo="percent",
                textposition="inside",
                insidetextorientation="radial",
                textfont=dict(size=12, color="#ffffff"),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Nilai: %{customdata[1]}<br>"
                    "Porsi: %{percent}<extra></extra>"
                ),
                customdata=list(zip(
                    lay_dist["layanan"],
                    [fmt_rp_id(v) for v in lay_dist["total"]],
                )),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8dced", size=12),
                legend=dict(
                    bgcolor="rgba(20,35,50,0.85)",
                    bordercolor="#2e4a6a",
                    borderwidth=1,
                    orientation="v",
                    x=1.01, y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=12),
                    itemsizing="constant",
                ),
                margin=dict(t=20, b=20, l=10, r=160),
                height=360,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_bill:
        st.markdown('<div class="section-header">🏆 Top 10 Biller</div>', unsafe_allow_html=True)
        top_biller = df_filt["biller_name"].value_counts().head(10).reset_index()
        top_biller.columns = ["biller", "jumlah"]
        top_biller = top_biller.sort_values("jumlah", ascending=True)
        fig_bill = px.bar(
            top_biller, x="jumlah", y="biller", orientation="h",
            color="jumlah",
            color_continuous_scale=["#1e4a6a", "#4fc3f7"],
            labels={"jumlah": "Jumlah Transaksi", "biller": "Biller"},
        )
        fig_bill.update_layout(
            coloraxis_showscale=False,
            height=360,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis","margin")},
            margin=dict(l=10, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#1e3048", zeroline=False, tickformat=",.0f"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
        )
        st.plotly_chart(fig_bill, use_container_width=True)


# ===========================================================================
# HALAMAN 2 : TREN & FORECAST
# ===========================================================================
elif page == "Tren & Forecast":
    st.markdown("""
    <h1 style="color:#e8f4fd; margin-bottom:4px;">📈 Tren & Forecast per Layanan</h1>
    <p style="color:#9cb3cc; margin-top:0;">Perbandingan nilai aktual vs prediksi LSTM dengan confidence interval ±5%</p>
    """, unsafe_allow_html=True)

    preds = load_predictions()

    if preds is None:
        st.warning("⚠️ Prediksi belum tersedia. Jalankan `03_evaluation.py` dulu.")
    else:
        lay_options = sorted(preds["layanan"].unique())

        col_sel, col_info = st.columns([1, 3])
        with col_sel:
            lay_pick = st.selectbox("🔗 Pilih Layanan", lay_options,
                                    key="lay_pick_tren")
        sub = preds[preds["layanan"] == lay_pick].sort_values("date")

        # Hitung CI ±5%
        sub = sub.copy()
        sub["ci_upper"] = sub["predicted"] * 1.05 / 1e6
        sub["ci_lower"] = sub["predicted"] * 0.95 / 1e6

        st.markdown(f'<div class="section-header">📉 Aktual vs Prediksi — {lay_pick}</div>',
                    unsafe_allow_html=True)

        fig_tren = go.Figure()

        # CI band
        fig_tren.add_trace(go.Scatter(
            x=pd.concat([sub["date"], sub["date"].iloc[::-1]]),
            y=pd.concat([sub["ci_upper"], sub["ci_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(255,183,77,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", name="±5% CI",
        ))
        fig_tren.add_trace(go.Scatter(
            x=sub["date"], y=sub["actual"] / 1e6,
            name="Aktual", mode="lines+markers",
            line=dict(color="#4fc3f7", width=2.5),
            marker=dict(size=5),
        ))
        fig_tren.add_trace(go.Scatter(
            x=sub["date"], y=sub["predicted"] / 1e6,
            name="Prediksi (LSTM)", mode="lines+markers",
            line=dict(color="#ffb74d", width=2, dash="dot"),
            marker=dict(size=5, symbol="x"),
        ))
        fig_tren.update_layout(
            yaxis_title="Nilai Transaksi (juta Rp)",
            xaxis_title="Tanggal",
            hovermode="x unified",
            height=420,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
            xaxis=dict(gridcolor="#1e3048", zeroline=False),
            yaxis=dict(gridcolor="#1e3048", zeroline=False, tickformat=",.0f"),
        )
        st.plotly_chart(fig_tren, use_container_width=True)

        # Tabel prediksi
        st.markdown('<div class="section-header">📋 Detail Prediksi</div>', unsafe_allow_html=True)
        tbl = sub[["date","layanan","actual","predicted"]].copy()
        tbl["actual"]    = tbl["actual"].map(fmt_rp_id)
        tbl["predicted"] = tbl["predicted"].map(fmt_rp_id)
        tbl.columns      = ["Tanggal", "Layanan", "Aktual", "Prediksi (LSTM)"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ===========================================================================
# HALAMAN 3 : DRILLDOWN
# ===========================================================================
elif page == "Drilldown":
    st.markdown("""
    <h1 style="color:#e8f4fd; margin-bottom:4px;">🔍 Drilldown Transaksi</h1>
    """, unsafe_allow_html=True)
    st.caption(f"Data tersaring: {fmt_int_id(len(df_filt))} transaksi")

    # ── KPI row ──
    c1, c2, c3 = st.columns(3)
    total_val = df_filt["trx_value"].sum()
    # Tampilkan dalam Juta Rp agar konsisten dengan Ringkasan yang pakai M (miliar)
    # → gunakan satuan yang sama: jika > 1 miliar pakai M, jika tidak pakai jt
    if total_val >= 1e9:
        total_str = fmt_idr(total_val, unit="m", decimals=3) + " M"
    else:
        total_str = fmt_idr(total_val, unit="jt", decimals=0) + " jt"

    c1.metric("💰 Total Nilai",   total_str)
    c2.metric("📈 Rata-rata",     fmt_rp_id(df_filt["trx_value"].mean()))
    c3.metric("📊 Median",        fmt_rp_id(df_filt["trx_value"].median()))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sampel Transaksi ──
    st.markdown('<div class="section-header">📄 Sampel Transaksi</div>', unsafe_allow_html=True)
    show_df = df_filt.head(500).copy()
    if "trx_value" in show_df.columns:
        show_df = show_df.rename(columns={"trx_value": "Nilai Transaksi"})
        show_df["Nilai Transaksi"] = show_df["Nilai Transaksi"].map(fmt_rp_id)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # ── Aktivitas per Jam ──
    st.markdown("---")
    st.markdown('<div class="section-header">🕐 Aktivitas per Jam (0–23)</div>', unsafe_allow_html=True)
    st.caption("Kapan nasabah paling aktif bertransaksi — berguna untuk perencanaan kapasitas server.")

    df_hour = df_filt.copy()
    df_hour["hour"] = df_hour["received_time"].dt.hour
    hourly = df_hour.groupby("hour").agg(
        jumlah_trx=("trx_value", "size"),
        total_nilai=("trx_value", "sum"),
    ).reindex(range(24), fill_value=0).reset_index()

    fig_jam = make_subplots(specs=[[{"secondary_y": True}]])
    fig_jam.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["jumlah_trx"],
        name="Jumlah Transaksi",
        marker=dict(color="#4fc3f7", opacity=0.8),
    ), secondary_y=False)
    fig_jam.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["total_nilai"] / 1e9,
        name="Total Nilai (miliar Rp)",
        mode="lines+markers",
        line=dict(color="#ffb74d", width=2.5),
        marker=dict(size=6),
    ), secondary_y=True)
    fig_jam.update_layout(
        height=380, hovermode="x unified",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
        xaxis=dict(gridcolor="#1e3048", zeroline=False, tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(gridcolor="#1e3048", zeroline=False, title="Jumlah Transaksi", tickformat=",.0f"),
        yaxis2=dict(title="Total Nilai (miliar Rp)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", zeroline=False, tickformat=",.1f"),
    )
    st.plotly_chart(fig_jam, use_container_width=True)

    # ── Aktivitas per Hari dalam Minggu ──
    st.markdown("---")
    st.markdown('<div class="section-header">📅 Aktivitas per Hari dalam Minggu</div>', unsafe_allow_html=True)
    st.caption("Apakah akhir pekan lebih sibuk? — mendukung keputusan pemasaran & staffing.")

    df_dow = df_filt.copy()
    df_dow["dow"] = df_dow["received_time"].dt.dayofweek
    dow_labels = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    weekly = df_dow.groupby("dow").agg(
        jumlah_trx=("trx_value", "size"),
        total_nilai=("trx_value", "sum"),
    ).reindex(range(7), fill_value=0).reset_index()
    weekly["nama_hari"] = dow_labels

    bar_colors = ["#4fc3f7"] * 5 + ["#ef5350"] * 2

    fig_dow = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dow.add_trace(go.Bar(
        x=weekly["nama_hari"], y=weekly["jumlah_trx"],
        name="Jumlah Transaksi",
        marker=dict(color=bar_colors, opacity=0.85),
    ), secondary_y=False)
    fig_dow.add_trace(go.Scatter(
        x=weekly["nama_hari"], y=weekly["total_nilai"] / 1e9,
        name="Total Nilai (miliar Rp)",
        mode="lines+markers",
        line=dict(color="#ffb74d", width=2.5),
        marker=dict(size=7),
    ), secondary_y=True)
    fig_dow.update_layout(
        height=360, hovermode="x unified",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
        xaxis=dict(gridcolor="#1e3048", zeroline=False),
        yaxis=dict(gridcolor="#1e3048", zeroline=False, title="Jumlah Transaksi", tickformat=",.0f"),
        yaxis2=dict(title="Total Nilai (miliar Rp)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", zeroline=False, tickformat=",.1f"),
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    # ── Top 20 Nasabah ──
    st.markdown("---")
    st.markdown('<div class="section-header">👤 Top 20 Nasabah Paling Aktif</div>', unsafe_allow_html=True)
    st.caption("Power users atau potensi anomali — nasabah dengan jumlah transaksi sangat tinggi layak diverifikasi.")

    top_cust = (df_filt.groupby("id_pelanggan")
                       .agg(jumlah_trx=("trx_value", "size"),
                            total_nilai=("trx_value", "sum"))
                       .sort_values("jumlah_trx", ascending=False)
                       .head(20)
                       .reset_index())

    def _mask(pid: str) -> str:
        pid = str(pid)
        if "|" in pid:
            head, tail = pid.split("|", 1)
            return f"{head}|{tail[:2]}***"
        return pid[:7] + "***"
    top_cust["id_masked"] = top_cust["id_pelanggan"].map(_mask)
    top_cust["label"] = [f"#{i+1:>2}  {m}" for i, m in enumerate(top_cust["id_masked"])]
    top_cust_sorted = top_cust.sort_values("jumlah_trx", ascending=True)

    fig_cust = px.bar(
        top_cust_sorted, x="jumlah_trx", y="label", orientation="h",
        color="jumlah_trx",
        color_continuous_scale=["#1a4a6a", "#4fc3f7"],
        labels={"jumlah_trx": "Jumlah Transaksi", "label": ""},
        text="jumlah_trx",
    )
    fig_cust.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                           textfont=dict(color="#c8dced", size=11))
    fig_cust.update_layout(
        coloraxis_showscale=False, height=560,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis","margin")},
        margin=dict(l=160, r=80, t=20, b=40),
        xaxis=dict(gridcolor="#1e3048", zeroline=False, tickformat=",.0f"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
    )
    st.plotly_chart(fig_cust, use_container_width=True)

    # Tabel pendamping
    show_cust = top_cust[["label", "jumlah_trx", "total_nilai"]].copy()
    show_cust["total_nilai"] = show_cust["total_nilai"].map(fmt_rp_id)
    show_cust["jumlah_trx"]  = show_cust["jumlah_trx"].map(fmt_int_id)
    show_cust.columns = ["Peringkat & ID", "Jumlah Transaksi", "Total Nilai"]
    st.dataframe(show_cust, use_container_width=True, hide_index=True)


# ===========================================================================
# HALAMAN 4 : ANOMALI
# ===========================================================================
elif page == "Anomali":
    st.markdown("""
    <h1 style="color:#e8f4fd; margin-bottom:4px;">⚠️ Deteksi Anomali</h1>
    <p style="color:#9cb3cc; margin-top:0;">Skor = |aktual − prediksi| / sigma residual. Skor > 3σ dianggap anomali.</p>
    """, unsafe_allow_html=True)

    preds = load_predictions()
    if preds is None:
        st.warning("⚠️ Prediksi belum tersedia. Jalankan `03_evaluation.py` dulu.")
    else:
        resid = preds["actual"] - preds["predicted"]
        sigma = resid.std() + 1e-9
        preds = preds.copy()
        preds["score"]      = (resid.abs() / sigma).round(2)
        preds["is_anomaly"] = preds["score"] > 3

        threshold = st.slider("🎚️ Ambang skor anomali",
                              min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        anomalies = preds[preds["score"] > threshold].sort_values("score", ascending=False)

        # ── KPI ──
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Titik",  fmt_int_id(len(preds)))
        c2.metric("⚠️ Anomali",      fmt_int_id(len(anomalies)))
        pct = len(anomalies) / len(preds) * 100 if len(preds) else 0
        c3.metric("📈 Persentase",   f"{pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Visualisasi Skor Anomali ──
        st.markdown('<div class="section-header">🔔 Visualisasi Skor Anomali</div>',
                    unsafe_allow_html=True)

        # Scatter plot skor per tanggal, warna merah=anomali, biru=normal
        fig_anom_scatter = go.Figure()

        normal_pts  = preds[~preds["is_anomaly"]]
        anomaly_pts = preds[preds["is_anomaly"]]

        fig_anom_scatter.add_trace(go.Scatter(
            x=normal_pts["date"], y=normal_pts["score"],
            mode="markers",
            name="Normal",
            marker=dict(color="#4fc3f7", size=6, opacity=0.6),
            hovertemplate="<b>%{x}</b><br>Skor: %{y:.2f}<extra>Normal</extra>",
        ))
        fig_anom_scatter.add_trace(go.Scatter(
            x=anomaly_pts["date"], y=anomaly_pts["score"],
            mode="markers",
            name="Anomali",
            marker=dict(color="#ef5350", size=12, symbol="diamond",
                        line=dict(color="#ff8a80", width=1.5)),
            hovertemplate="<b>%{x}</b><br>Skor: %{y:.2f}<extra>ANOMALI</extra>",
        ))
        # Garis threshold
        date_min_p = preds["date"].min()
        date_max_p = preds["date"].max()
        fig_anom_scatter.add_shape(
            type="line",
            x0=date_min_p, x1=date_max_p,
            y0=threshold, y1=threshold,
            line=dict(color="#ffb74d", width=2, dash="dash"),
        )
        fig_anom_scatter.add_annotation(
            x=date_max_p, y=threshold,
            text=f"Ambang {threshold:.1f}σ",
            showarrow=False,
            font=dict(color="#ffb74d", size=11),
            xanchor="right", yanchor="bottom",
        )
        fig_anom_scatter.update_layout(
            height=380, hovermode="closest",
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
            xaxis=dict(gridcolor="#1e3048", zeroline=False, title="Tanggal"),
            yaxis=dict(gridcolor="#1e3048", zeroline=False, title="Skor Anomali (σ)"),
        )
        st.plotly_chart(fig_anom_scatter, use_container_width=True)

        # ── Daftar Anomali ──
        st.markdown('<div class="section-header">📋 Daftar Anomali</div>', unsafe_allow_html=True)
        view = anomalies[["date", "layanan", "actual", "predicted", "score"]].copy()
        view["actual"]    = view["actual"].map(fmt_rp_id)
        view["predicted"] = view["predicted"].map(fmt_rp_id)
        view.columns = ["Tanggal", "Layanan", "Aktual", "Prediksi", "Skor (σ)"]
        st.dataframe(view, use_container_width=True, hide_index=True)


# ===========================================================================
# HALAMAN 5 : EVALUASI MODEL
# ===========================================================================
elif page == "Evaluasi Model":
    st.markdown("""
    <h1 style="color:#e8f4fd; margin-bottom:4px;">✏️ Evaluasi Model LSTM</h1>
    <p style="color:#9cb3cc; margin-top:0;">Metrik performa model secara global dan per layanan, beserta scatter plot aktual vs prediksi.</p>
    """, unsafe_allow_html=True)

    metrics = load_metrics()
    per_lay = load_per_layanan_metrics()
    preds   = load_predictions()

    if metrics is None:
        st.warning("⚠️ Hasil evaluasi belum tersedia. Jalankan `03_evaluation.py` dulu.")
    else:
        # ── Metrik Global (2×2 cards) ──
        st.markdown('<div class="section-header">🏆 Metrik Global</div>', unsafe_allow_html=True)
        m = metrics["overall"]
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        # Format angka dengan titik standar Indonesia
        mae_str  = "Rp " + f"{m['MAE']:,.0f}".replace(",", ".")
        rmse_str = "Rp " + f"{m['RMSE']:,.0f}".replace(",", ".")
        mape_str = f"{m['MAPE']:.2f}%"
        mse_t    = m['MSE'] / 1e12
        mse_str  = f"{mse_t:,.0f}".replace(",", ".") + " T Rp²"

        r1c1.metric("📉 MAE",  mae_str,  "Mean Absolute Error")
        r1c2.metric("📊 RMSE", rmse_str, "Root Mean Squared Error")
        r1c3.metric("🎯 MAPE", mape_str, "Mean Absolute Percentage Error")
        r1c4.metric("⚡ MSE",  mse_str,  "Mean Squared Error")

        st.caption("ℹ️ Satuan MSE adalah Rp² (Rupiah kuadrat). T = Triliun. "
                   "RMSE adalah akar MSE dengan satuan Rupiah yang lebih intuitif.")

        # ── Konfigurasi Terbaik ──
        st.markdown('<div class="section-header">⚙️ Konfigurasi Terbaik (Grid Search)</div>',
                    unsafe_allow_html=True)
        st.json(metrics["best_config"])

        # ── Metrik per Layanan ──
        if per_lay is not None:
            st.markdown('<div class="section-header">📋 Metrik per Layanan</div>',
                        unsafe_allow_html=True)

            # Bar chart MAPE per layanan
            per_lay_chart = per_lay.copy()
            per_lay_chart["color"] = per_lay_chart["MAPE"].apply(
                lambda x: "#ef5350" if x > 100 else ("#ffb74d" if x > 30 else "#81c784")
            )
            per_lay_chart["mape_pct"] = per_lay_chart["MAPE"].map(lambda x: f"{x:.1f}%")

            fig_mape = go.Figure(go.Bar(
                x=per_lay_chart["layanan"],
                y=per_lay_chart["MAPE"],
                marker=dict(color=per_lay_chart["color"], opacity=0.85),
                text=per_lay_chart["mape_pct"],
                textposition="outside",
                textfont=dict(color="#c8dced"),
                hovertemplate="<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>",
            ))
            fig_mape.update_layout(
                yaxis_title="MAPE (%)", xaxis_title="Layanan",
                height=340,
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
                xaxis=dict(gridcolor="#1e3048", zeroline=False),
                yaxis=dict(gridcolor="#1e3048", zeroline=False),
            )
            st.plotly_chart(fig_mape, use_container_width=True)

            # Tabel metrik per layanan — format titik standar ID
            show = per_lay.copy()
            show["MAE"]  = show["MAE"].map(lambda x: "Rp " + f"{x:,.0f}".replace(",","."))
            show["RMSE"] = show["RMSE"].map(lambda x: "Rp " + f"{x:,.0f}".replace(",","."))
            show["MAPE"] = show["MAPE"].map(lambda x: f"{x:.2f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)

        # ── Scatter Aktual vs Prediksi ──
        if preds is not None:
            st.markdown('<div class="section-header">🔵 Scatter — Aktual vs Prediksi</div>',
                        unsafe_allow_html=True)

            # Hitung skor anomali untuk pewarnaan scatter
            resid_s      = preds["actual"] - preds["predicted"]
            sigma_s      = resid_s.std() + 1e-9
            preds_s      = preds.copy()
            preds_s["score"] = (resid_s.abs() / sigma_s).round(2)
            preds_s["is_outlier"] = preds_s["score"] > 3

            normal_s  = preds_s[~preds_s["is_outlier"]]
            outlier_s = preds_s[preds_s["is_outlier"]]

            lo = min(preds_s["actual"].min(), preds_s["predicted"].min()) / 1e6
            hi = max(preds_s["actual"].max(), preds_s["predicted"].max()) / 1e6

            fig_sc = go.Figure()

            # Titik normal
            fig_sc.add_trace(go.Scatter(
                x=normal_s["actual"] / 1e6,
                y=normal_s["predicted"] / 1e6,
                mode="markers",
                name="Data Point",
                marker=dict(color="#4fc3f7", size=6, opacity=0.6),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Aktual: Rp %{x:,.1f} jt<br>"
                    "Prediksi: Rp %{y:,.1f} jt<extra>Normal</extra>"
                ),
                customdata=normal_s[["layanan"]].values,
            ))

            # Titik anomali — warna merah + simbol berbeda
            if len(outlier_s) > 0:
                fig_sc.add_trace(go.Scatter(
                    x=outlier_s["actual"] / 1e6,
                    y=outlier_s["predicted"] / 1e6,
                    mode="markers",
                    name="Anomali (skor > 3σ)",
                    marker=dict(color="#ef5350", size=10, symbol="diamond",
                                line=dict(color="#ff8a80", width=1.5), opacity=0.9),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Aktual: Rp %{x:,.1f} jt<br>"
                        "Prediksi: Rp %{y:,.1f} jt<br>"
                        "Skor: %{customdata[1]:.2f}σ<extra>ANOMALI</extra>"
                    ),
                    customdata=outlier_s[["layanan","score"]].values,
                ))

            # Garis y = x (ideal)
            fig_sc.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines", name="y = x (ideal)",
                line=dict(color="#ff7043", width=2, dash="dash"),
            ))

            fig_sc.update_layout(
                xaxis_title="Aktual (juta Rp)",
                yaxis_title="Prediksi (juta Rp)",
                height=500,
                hovermode="closest",
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
                xaxis=dict(gridcolor="#1e3048", zeroline=False, tickformat=",.0f"),
                yaxis=dict(gridcolor="#1e3048", zeroline=False, tickformat=",.0f"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
