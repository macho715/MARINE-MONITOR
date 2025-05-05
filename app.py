import logging
import pandas as pd
import altair as alt
import streamlit as st

# 내부 모듈
from fetcher import fetch_and_process_data, LAT, LON

# ──────────────────────────────────────────────
# Page config
st.set_page_config(page_title="Marine Weather Monitor", layout="wide")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

st.title("🌊 Marine Weather Monitor (7‑Day Forecast)")
st.caption(f"Monitoring conditions near Lat: {LAT}, Lon: {LON}")

# ──────────────────────────────────────────────
# Caching wrapper (10 min TTL)
@st.cache_data(ttl=600) # Note: Previous step set TTL to 1800, this code uses 600. Keeping 600 for now as per user code.
def get_data(tide_key: str | None = None):
    logging.info("Cache miss / expired → fetching new data…")
    return fetch_and_process_data(tide_key)

# Manual refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()

# Secrets (WorldTides API optional)
TIDE_KEY = st.secrets.get("TIDE_KEY", None)
if not TIDE_KEY:
     st.sidebar.warning("WorldTides API Key (TIDE_KEY) not found in secrets. Tide data will not be fetched.", icon="🔑") # Keep sidebar warning

# ──────────────────────────────────────────────
# Data fetch
with st.spinner("Fetching latest 7‑day forecast data…"):
    df, trend_rows, daily_rows, err = get_data(TIDE_KEY)

if err:
    st.error(err)
    st.stop()

if df.empty:
    st.warning("⚠️ No data returned from API.")
    st.stop()

# ──────────────────────────────────────────────
# Forecast summary tables
st.subheader("Forecast Summary & Sailing Decision")

trend_df = pd.DataFrame(
    trend_rows,
    columns=["Day (LT)", "모드", "파고", "풍속", "위험등급", "작업 권고"]
)
if not trend_df.empty:
    st.dataframe(trend_df, hide_index=True, use_container_width=True)
else:
    st.info("Trend summary not available.")

st.markdown("---")

st.subheader("7‑Day Forecast Trend Chart")

# ──────────────────────────────────────────────
# Interactive layer toggle
col1, col2, col3 = st.columns(3)
with col1:
    show_wave = st.checkbox("🌊 SWH (m)", value=True)
with col2:
    show_wind = st.checkbox("💨 Wind (kt)", value=True)
with col3:
    # Only show tide checkbox if tide data is available
    show_tide = False
    if 'tide_m' in df.columns and df['tide_m'].notna().any():
         show_tide = st.checkbox("🌑 Tide (m)", value=False)
st.markdown("<hr style='margin-top:0; margin-bottom:1rem;'>", unsafe_allow_html=True)

base = alt.Chart(df).encode(x=alt.X("time:T", title="Time (GST)"))
layers: list[alt.Chart] = []

if show_wave:
    layers.append(
        base.mark_line(color="#1f77b4", strokeWidth=1.5).encode(
            y=alt.Y("wave_m:Q", title="SWH (m)", axis=alt.Axis(titleColor="#1f77b4")) # Added titleColor
            ,tooltip=['time:T', alt.Tooltip('wave_m:Q', format='.1f', title='SWH (m)')] # Added tooltip
        )
    )

if show_wind:
    layers.append(
        base.mark_line(color="#ff7f0e", strokeWidth=1.5, strokeDash=[4,3]).encode(
            y=alt.Y("wind_kt:Q", title="Wind (kt)", axis=alt.Axis(titleColor="#ff7f0e")) # Added titleColor
             ,tooltip=['time:T', alt.Tooltip('wind_kt:Q', format='.0f', title='Wind (kt)')] # Added tooltip
        )
    )

if show_tide and "tide_m" in df.columns:
    layers.append(
        base.mark_line(color="#268bd2", strokeDash=[2,2]).encode(
            y=alt.Y("tide_m:Q", title="Tide (m)", axis=alt.Axis(titleColor="#268bd2")) # Added titleColor
             ,tooltip=['time:T', alt.Tooltip('tide_m:Q', format='.1f', title='Tide (m)')] # Added tooltip
        )
    )

# Sail limits (always displayed for context)
# Assuming THR_WAVE=2.0, THR_WIND=12.0 are the values
THR_WAVE_VAL = 2.0 
THR_WIND_VAL = 12.0
limits_df = pd.DataFrame({
    "y": [THR_WAVE_VAL, THR_WIND_VAL],
    "label": [f"Sail SWH Limit ({THR_WAVE_VAL:.1f} m)", f"Sail Wind Limit ({THR_WIND_VAL:.0f} kt)"],
    "c": ["red", "orange"]
})
limits = alt.Chart(limits_df).mark_rule(strokeDash=[4,4]).encode(
    y="y:Q", 
    color=alt.Color("c:N", scale=None), # Use direct color names
    tooltip=['label:N'] 
)

if layers:
    chart = alt.layer(*layers, limits).resolve_scale(y="independent").properties(height=320)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("👆 위 체크박스로 표시할 항목을 선택하세요.")

# ──────────────────────────────────────────────
# Daily sailing table
st.markdown("---")
st.markdown("**일별 항해 가능 여부 (Al Ghallan→Abu Dhabi)**")

daily_df = pd.DataFrame(
    daily_rows,
    columns=["Date", "00‑12 LT Wind / Wave", "12‑24 LT Wind / Wave", "Risk", "Decision", "참고"]
)
if not daily_df.empty:
    st.dataframe(daily_df, hide_index=True, use_container_width=True)
else:
    st.info("Daily sailing table not available.")

# ──────────────────────────────────────────────
# Raw data (optional)
with st.expander("Raw hourly data (debug)"):
    if not df.empty:
         df_vis = df.copy()
         # Select and format columns for display
         display_cols = ['time', 'wave_m', 'wind_kt']
         if 'tide_m' in df_vis.columns and df_vis['tide_m'].notna().any(): # Check if tide exists and has data
              display_cols.append('tide_m')
         if 'wave_pred' in df_vis.columns and df_vis['wave_pred'].notna().any(): # Check if forecast exists and has data
              display_cols.append('wave_pred')
         df_vis = df_vis[display_cols] # Keep only relevant cols for display
         
         df_vis["time"] = pd.to_datetime(df_vis["time"]).dt.strftime("%Y-%m-%d %H:%M")
         for col in df_vis.columns:
             if col != 'time' and pd.api.types.is_numeric_dtype(df_vis[col]):
                 df_vis[col] = df_vis[col].round(1) # Round numeric cols
             if col == 'wind_kt': # Specific formatting for wind knots
                 df_vis[col] = df_vis[col].round(0).astype('Int64') # Use nullable Int
         
         st.dataframe(df_vis, hide_index=True, use_container_width=True)
         
         # Add CSV download button
         csv_data = df.to_csv(index=False).encode('utf-8')
         st.download_button(
             label="📥 Download Raw Data as CSV",
             data=csv_data,
             file_name='7day_forecast_raw.csv',
             mime='text/csv',
         )
    else:
         st.warning("Raw hourly data is not available.")

st.markdown("---")
st.caption("Data © Open‑Meteo & WorldTides | Dashboard generated with Streamlit") 