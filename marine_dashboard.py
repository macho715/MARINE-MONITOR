import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import datetime as dt
import numpy as np # Added for potential use, though limits might use it indirectly

# fetcher.py에서 주요 함수들을 가져옵니다.
# fetch_and_process_data 대신 get_combined_forecast_data를 사용하고 반환값 변경에 주의
from fetcher import get_combined_forecast_data, assess_operational_risk, LAT, LON 

TIDE_LIMIT = 1.5 # m
# WAVE_LIMIT_FT = 6.0 # ft (report.py와 동일 기준) - NCM 값과 비교를 위해 m로 변환하여 사용
WAVE_LIMIT_M = 1.8288 # 6ft를 미터로 변환 (NCM 기준과 통일)
WIND_LIMIT_KT = 20.0 # knots (report.py와 동일 기준)

# --- Page Configuration ---
st.set_page_config(page_title="Marine Forecast Dashboard", layout="wide")

# --- Helper Functions ---
def get_available_dates_for_selection(days_range=30):
    base = dt.datetime.today()
    date_list = [(base - dt.timedelta(days=x)).strftime("%Y%m%d") for x in range(days_range)]
    return sorted(date_list, reverse=True)

# 새로운 preprocess 함수 정의 (사용자 제안 기반)
def preprocess(df):
    if df.empty:
        st.info("Cannot preprocess empty DataFrame.")
        return df
        
    required_cols_preprocess = ['time', 'go_nogo', 'risk_level']
    if not all(col in df.columns for col in required_cols_preprocess):
        st.warning(f"Preprocessing requires columns: {required_cols_preprocess}. Skipping summary and timeline.")
        return df

    def label_status(r):
        go_nogo_status = r.get('go_nogo')
        risk_level_status = r.get('risk_level')
        if go_nogo_status == "GO": return "Go"
        if risk_level_status == "Med":
            if go_nogo_status != "NO-GO": return "Delay"
        if go_nogo_status == "NO-GO": return "NoGo"
        return "Unknown" 
        
    df["status"] = df.apply(label_status, axis=1)
    hours_per_row = 1 
    summary = df['status'].value_counts().reindex(["Go","Delay","NoGo", "Unknown"], fill_value=0)
    go_hr = summary.get("Go", 0) * hours_per_row
    delay_hr = summary.get("Delay", 0) * hours_per_row
    nogo_hr = summary.get("NoGo", 0) * hours_per_row
    st.subheader("Forecast Period Summary (Approx. Hours)")
    cols = st.columns(3)
    cols[0].metric("✅ Go Hours", f"{go_hr} hrs")
    cols[1].metric("🟠 Delay Hours", f"{delay_hr} hrs")
    cols[2].metric("❌ No-Go Hours", f"{nogo_hr} hrs")

    NOGO_ALERT_THRESHOLD_HOURS = 120
    if nogo_hr >= NOGO_ALERT_THRESHOLD_HOURS:
        st.sidebar.error(f"⚠️ {int(nogo_hr/24)}일 이상 운항 불가(No-Go) 상태가 지속됩니다. 일정 재검토가 필요합니다.")

    delay_windows = []
    in_delay_window = False
    start_time = None
    min_delay_duration_hours = 3
    if 'status' in df.columns and 'time' in df.columns:
        for index, row in df.iterrows():
            current_status = row['status']
            current_time = row['time']
            if current_status == 'Delay' and not in_delay_window:
                in_delay_window = True
                start_time = current_time
            elif current_status != 'Delay' and in_delay_window:
                in_delay_window = False
                if start_time:
                    end_time = current_time 
                    duration = (end_time - start_time).total_seconds() / 3600
                    if duration >= min_delay_duration_hours:
                        delay_windows.append((start_time.strftime("%m-%d %H:%M"), end_time.strftime("%m-%d %H:%M")))
                    start_time = None 
        if in_delay_window and start_time:
             end_time = df['time'].iloc[-1] + pd.Timedelta(hours=hours_per_row)
             duration = (end_time - start_time).total_seconds() / 3600
             if duration >= min_delay_duration_hours:
                  delay_windows.append((start_time.strftime("%m-%d %H:%M"), end_time.strftime("%H:%M")))
    if delay_windows:
        st.info(f"🟠 **Suggested Delay Windows (≥ {min_delay_duration_hours} hrs):**")
        for start, end in delay_windows:
            st.write(f" - From {start} to {end}")
    else:
        st.info("No significant delay windows found.")

    st.subheader("Operational Status Timeline")
    color_scale = alt.Scale(domain=["Go","Delay","NoGo", "Unknown"], range=["#2ca02c","#ff7f0e","#d62728", "#cccccc"])
    if not df.empty and 'status' in df.columns and pd.notna(df['status']).any():
        timeline_chart = (alt.Chart(df).mark_rule(size=8).encode(
                x=alt.X('time:T', title='Time (UTC)', axis=alt.Axis(grid=False)),
                y=alt.value(10),
                color=alt.Color('status:N', scale=color_scale, title='Status',
                                legend=alt.Legend(title="Status", orient="right", values=["Go","Delay","NoGo", "Unknown"])),
                tooltip=[alt.Tooltip('time:T', title='Time'), 'status:N', 'go_nogo:N', 'risk_reason:N']
            ).properties(height=50))
        st.altair_chart(timeline_chart, use_container_width=True)
    else:
        st.info("Timeline 차트를 표시할 유효한 상태 데이터가 없습니다.")
    return df

def find_tide_windows(df_with_time_index_and_col, limit=1.5, min_hours=3):
    safe = df_with_time_index_and_col[df_with_time_index_and_col["tide_m"] >= limit]
    if safe.empty:
        return pd.DataFrame(columns=["min", "max", "duration_h"])
    grp  = (safe.index.to_series().diff() != pd.Timedelta("1h")).cumsum()
    windows = safe.groupby(grp)["time"].agg(["min", "max", "count"])
    windows = windows[windows["count"] >= min_hours]
    windows["duration_h"] = windows["count"]
    return windows[["min", "max", "duration_h"]]

# --- Sidebar ---
st.sidebar.title("Marine Dashboard")

tide_api_key = None
try:
    tide_api_key = st.secrets.get("TIDE_KEY")
    if not tide_api_key:
        st.sidebar.error("TIDE_KEY가 Streamlit Secrets에 설정되지 않았습니다. 조수 데이터를 가져올 수 없습니다.")
except Exception as e:
    st.sidebar.error(f"Secrets 파일을 읽는 중 오류 발생: {e}. TIDE_KEY를 확인해주세요.")

available_dates = get_available_dates_for_selection()
if not available_dates:
    st.sidebar.warning("선택 가능한 날짜가 없습니다.")
    st.warning("데이터를 표시할 수 없습니다.")
    st.stop()
selected_date_str = st.sidebar.selectbox("Select Start Date", available_dates)

# --- Main Content ---
df_hourly_forecast = pd.DataFrame() # Initialize hourly forecast DataFrame
df_daily_ncm_comparison = pd.DataFrame() # Initialize NCM comparison DataFrame
error_message_combined = None

if not tide_api_key and selected_date_str:
    st.error("TIDE_KEY가 없어 조수 데이터를 포함한 예보를 생성할 수 없습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

if selected_date_str and tide_api_key:
    with st.spinner(f"{selected_date_str} 기준 예보 및 NCM 데이터를 가져오는 중..."):
        # get_combined_forecast_data는 이제 (hourly_df, daily_ncm_df, error_msg)를 반환
        df_hourly_forecast, df_daily_ncm_comparison, error_message_combined = get_combined_forecast_data(
            selected_date_str=selected_date_str,
            tide_api_key=tide_api_key,
            # days_to_fetch는 get_combined_forecast_data 내부의 기본값(DAYS=7)을 사용
        )

    if error_message_combined:
        st.error(f"데이터 로딩 중 오류 발생: {error_message_combined}")
        # 오류 발생 시에도 부분 데이터가 있을 수 있으므로, 일단 진행하고 아래에서 empty 체크
    
    if df_hourly_forecast.empty:
        st.warning(f"{selected_date_str}에 대한 시간별 예보 데이터를 가져오지 못했거나 데이터가 없습니다.")
        # NCM 데이터만 있을 수도 있으므로 st.stop()은 아직 하지 않음

    # df_hourly_forecast가 있어도 preprocess 전에 risk assessment가 이미 fetcher에서 수행됨.
    # preprocess는 주로 status 라벨링 및 요약/타임라인 UI를 담당.
else:
    if not selected_date_str:
        st.info("시작 날짜를 선택해주세요.")
    st.stop() # 날짜 선택 없으면 중단

# --- NCM vs API Sidebar Alert ---
if not df_daily_ncm_comparison.empty:
    # Check for significant discrepancies if NCM data is available
    # Ensure columns exist before trying to access them
    wave_gap_col_exists = 'wave_gap_%' in df_daily_ncm_comparison.columns
    wind_gap_col_exists = 'wind_gap_%' in df_daily_ncm_comparison.columns
    
    max_wave_gap = 0
    if wave_gap_col_exists and df_daily_ncm_comparison['wave_gap_%'].notna().any():
        max_wave_gap = df_daily_ncm_comparison["wave_gap_%"].abs().max()
        
    max_wind_gap = 0
    if wind_gap_col_exists and df_daily_ncm_comparison['wind_gap_%'].notna().any():
        max_wind_gap = df_daily_ncm_comparison["wind_gap_%"].abs().max()

    if pd.notna(max_wave_gap) and max_wave_gap > 20 or pd.notna(max_wind_gap) and max_wind_gap > 20:
        alert_message = "⚠️ API↔NCM 차이 >20% (단위/좌표 점검 필요):"
        if pd.notna(max_wave_gap) and max_wave_gap > 20:
            alert_message += f" Wave Max Gap: {max_wave_gap:.1f}%"
        if pd.notna(max_wind_gap) and max_wind_gap > 20:
            alert_message += f" Wind Max Gap: {max_wind_gap:.1f}%"
        st.sidebar.error(alert_message)
    elif (wave_gap_col_exists and df_daily_ncm_comparison['wave_gap_%'].notna().any()) or \
         (wind_gap_col_exists and df_daily_ncm_comparison['wind_gap_%'].notna().any()):
        # NCM 데이터가 있고, 큰 차이가 없는 경우
        st.sidebar.info("✅ API↔NCM 오차 ±20% 이내 (해당일 데이터 기준).")
    # else: NCM 데이터가 없거나 gap 계산이 안된 경우는 별도 메시지 없음 (이미 fetcher에서 로그)
else:
    st.sidebar.info("NCM 비교 데이터가 없습니다.")


if df_hourly_forecast.empty: # 최종 df 확인 후 중단 결정
    st.warning("처리할 시간별 예보 데이터가 없습니다. 대시보드를 표시할 수 없습니다.")
    st.stop()

# --- Preprocess 데이터 (Status 추가, Summary 표시, Timeline 차트 표시) ---
# df_hourly_forecast는 이미 risk assessment가 완료된 상태로 전달됨
df_processed_hourly = preprocess(df_hourly_forecast) 

# Sidebar Alert for Low Tide (기존 로직 유지, df_processed_hourly 사용)
if "tide_m" in df_processed_hourly.columns and pd.notna(df_processed_hourly["tide_m"]).any():
    min_tide_forecast_period = df_processed_hourly["tide_m"].min()
    if pd.notna(min_tide_forecast_period) and min_tide_forecast_period < TIDE_LIMIT:
        st.sidebar.error(f"⚠️ 예보 기간 중 최저 조위가 {TIDE_LIMIT}m 미만({min_tide_forecast_period:.2f}m)으로 예측됩니다!")
elif "tide_m" not in df_processed_hourly.columns:
    st.sidebar.info("조수 데이터 컬럼('tide_m')이 없습니다.")
else:
    st.sidebar.info("조수 데이터가 유효하지 않습니다 (모든 값이 NA).")

start_dt_display = pd.to_datetime(df_processed_hourly['time'].min()).strftime('%Y-%m-%d') if 'time' in df_processed_hourly and not df_processed_hourly.empty else selected_date_str
st.title(f"Al Ghallan Forecast Dashboard – {start_dt_display} 부터")
st.caption(f"데이터 소스: fetcher.py (위도: {LAT:.3f}, 경도: {LON:.3f})")
if "tide_m" in df_processed_hourly.columns and pd.notna(df_processed_hourly["tide_m"]).any():
    st.caption("조수 데이터 (WorldTides API) 통합됨. NCM 공식 예보와 비교 제공.")

# --- NCM Official vs API Max (Daily) Table ---
st.subheader("NCM Official vs API Max Forecast (Daily Comparison)")
if not df_daily_ncm_comparison.empty:
    # 컬럼 이름 및 순서 사용자 정의에 맞게 조정
    display_ncm_cols = {
        'date': 'Date',
        'wave_ft_max_off': 'NCM Wave (ft)',
        # 'wave_m_max_api': 'API Wave (m)', # API Wave는 ft로 변환된 값을 사용
        'wave_ft_max_api': 'API Wave (ft)',
        'wave_gap_%': 'Wave Gap (%)',
        'wind_kn_max_off': 'NCM Wind (kt)',
        'wind_kt_max_api': 'API Wind (kt)',
        'wind_gap_%': 'Wind Gap (%)'
    }
    # 보여줄 컬럼들 (순서대로)
    ordered_cols_to_display = ['date', 'wave_ft_max_off', 'wave_ft_max_api', 'wave_gap_%', 'wind_kn_max_off', 'wind_kt_max_api', 'wind_gap_%']
    
    # 실제 df_daily_ncm_comparison에 있는 컬럼만 선택
    existing_cols_to_display = [col for col in ordered_cols_to_display if col in df_daily_ncm_comparison.columns]
    df_display_ncm = df_daily_ncm_comparison[existing_cols_to_display].copy()
    
    # 날짜 형식 변경
    if 'date' in df_display_ncm.columns:
        df_display_ncm['date'] = pd.to_datetime(df_display_ncm['date']).dt.strftime('%Y-%m-%d')
        
    df_display_ncm.rename(columns=display_ncm_cols, inplace=True)
    st.dataframe(df_display_ncm, use_container_width=True)
else:
    st.info("NCM 비교 데이터를 가져올 수 없거나 해당 기간에 데이터가 없습니다.")

st.subheader("시간별 예보 데이터 및 위험도 평가")
# display_columns는 df_processed_hourly 기준으로
display_columns_hourly = ['time', 'wave_m', 'wind_kt', 'tide_m', 'wave_pred', 'risk_level', 'go_nogo', 'risk_reason']
display_df_hourly = df_processed_hourly[[col for col in display_columns_hourly if col in df_processed_hourly.columns]].copy()
if 'time' in display_df_hourly.columns:
    display_df_hourly['time'] = pd.to_datetime(display_df_hourly['time']).dt.strftime('%Y-%m-%d %H:%M')
st.dataframe(display_df_hourly)

st.subheader("예보 차트")
limits = {
    'wave_m': WAVE_LIMIT_M, # NCM 파고 한계 (m 단위)
    'wind_kt': WIND_LIMIT_KT,
    'tide_m': TIDE_LIMIT
}
limit_colors = {
    'wave_m': 'red',
    'wind_kt': 'orange',
    'tide_m': 'blue'
}

chart_metrics = []
if 'wave_m' in df_processed_hourly.columns and pd.notna(df_processed_hourly['wave_m']).any(): chart_metrics.append('wave_m')
if 'wind_kt' in df_processed_hourly.columns and pd.notna(df_processed_hourly['wind_kt']).any(): chart_metrics.append('wind_kt')
# wave_pred는 현재 fetcher의 placeholder API 생성 로직에는 없음. 추가 시 아래 주석 해제
# if 'wave_pred' in df_processed_hourly.columns and pd.notna(df_processed_hourly['wave_pred']).any(): chart_metrics.append('wave_pred')

show_tide_default = 'tide_m' in df_processed_hourly.columns and pd.notna(df_processed_hourly['tide_m']).any()
show_tide = st.checkbox("차트에 조수 데이터 포함", value=show_tide_default, key="show_tide_checkbox_main")

if show_tide and 'tide_m' in df_processed_hourly.columns and pd.notna(df_processed_hourly['tide_m']).any():
    if 'tide_m' not in chart_metrics: chart_metrics.append('tide_m')
elif 'tide_m' in chart_metrics:
    chart_metrics.remove('tide_m')

if not chart_metrics:
    st.warning("차트에 표시할 수 있는 예보 데이터가 없습니다.")
else:
    if not df_processed_hourly.empty and 'time' in df_processed_hourly.columns and all(metric in df_processed_hourly.columns for metric in chart_metrics):
        line_chart_data = df_processed_hourly[['time'] + chart_metrics].melt(id_vars=["time"], var_name="Metric", value_name="Value")
        line_chart_data.dropna(subset=['Value'], inplace=True)
        if not line_chart_data.empty:
            base_chart = alt.Chart(line_chart_data).mark_line(point=True).encode(
                x=alt.X("time:T", title="Time (UTC)"),
                y=alt.Y("Value:Q", title="Value", scale=alt.Scale(zero=False)),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=["time:T", "Metric:N", alt.Tooltip("Value:Q", format=".2f")]
            ).properties(height=350).interactive()
            
            layers = [base_chart]
            
            for metric in chart_metrics:
                if metric in limits:
                    limit_value = limits[metric]
                    limit_color = limit_colors.get(metric, 'red')
                    limit_df = pd.DataFrame({'limit': [limit_value]})
                    rule = alt.Chart(limit_df).mark_rule(color=limit_color, strokeDash=[4,4], size=1.5).encode(y='limit:Q')
                    layers.append(rule)
            
            # --- NCM Data Overlay on Chart ---
            if not df_daily_ncm_comparison.empty and 'date' in df_daily_ncm_comparison.columns and \
               'wave_ft_max_off' in df_daily_ncm_comparison.columns and 'wave_m' in chart_metrics: # Only add if wave_m is plotted
                
                ncm_wave_data_for_chart = df_daily_ncm_comparison[
                    ['date', 'wave_ft_max_off']
                ].copy()
                ncm_wave_data_for_chart.dropna(subset=['wave_ft_max_off'], inplace=True)
                
                if not ncm_wave_data_for_chart.empty:
                    ncm_wave_data_for_chart['ncm_wave_m'] = ncm_wave_data_for_chart['wave_ft_max_off'] / 3.28084 # Convert NCM ft to m
                    # Create datetime objects for Altair chart (e.g., noon of the day for plotting a daily mark)
                    ncm_wave_data_for_chart['time_marker'] = pd.to_datetime(ncm_wave_data_for_chart['date']) + pd.Timedelta(hours=12)

                    # NCM Max Wave rule (horizontal line for each day if NCM data exists)
                    # This creates a step-like line if NCM max changes day by day.
                    # We need to ensure this is plotted correctly on the time axis.
                    # Creating segments for each day for the NCM line.
                    ncm_chart_lines = []
                    for _, ncm_row in ncm_wave_data_for_chart.iterrows():
                        day_start = pd.to_datetime(ncm_row['date'])
                        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # End of the day
                        ncm_val = ncm_row['ncm_wave_m']
                        
                        # Create a small df for this day's NCM line segment
                        ncm_segment_df = pd.DataFrame({
                            'time_from': [day_start],
                            'time_to': [day_end],
                            'ncm_limit': [ncm_val]
                        })
                        
                        ncm_line_segment = alt.Chart(ncm_segment_df).mark_rule(color="#f39c12", strokeDash=[2,2], size=1.8).encode(
                            x='time_from:T',
                            x2='time_to:T',
                            y='ncm_limit:Q'
                        ).properties(title="NCM Max Wave (Orange Dash)")
                        ncm_chart_lines.append(ncm_line_segment)
                    
                    if ncm_chart_lines:
                        # Add a dummy layer for the legend entry if needed, or ensure title is descriptive
                        # For simplicity, we rely on chart title/description for now for what the orange line is.
                        # layers.extend(ncm_chart_lines) # Add all segments
                        # Let's try a single layer approach if Altair handles it well with a step chart for daily NCM values
                        # This might be simpler than multiple rule segments
                        
                        # Create data for step chart for NCM daily max wave
                        ncm_step_data = []
                        for i, ncm_row in ncm_wave_data_for_chart.iterrows():
                            day_start = pd.to_datetime(ncm_row['date'])
                            day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                            ncm_val_m = ncm_row['ncm_wave_m']
                            ncm_step_data.append({'time': day_start, 'NCM Wave (m)': ncm_val_m})
                            ncm_step_data.append({'time': day_end, 'NCM Wave (m)': ncm_val_m})
                        
                        if ncm_step_data:
                            ncm_step_df = pd.DataFrame(ncm_step_data)
                            ncm_wave_line = alt.Chart(ncm_step_df).mark_line(color="#f39c12", interpolate='step-after', strokeDash=[3,3], size=2).encode(
                                x='time:T',
                                y=alt.Y('NCM Wave (m):Q', title="NCM Max Wave (m)")
                            )
                            layers.append(ncm_wave_line)

            final_layered_chart = alt.layer(*layers).resolve_scale(y='independent' if len(chart_metrics) > 1 else 'shared')
            st.altair_chart(final_layered_chart, use_container_width=True)
        else:
            st.info("선택된 측정 항목에 대한 유효한 데이터가 없어 차트를 표시할 수 없습니다.")
    else:
        st.warning("차트 표시에 필요한 'time' 컬럼 또는 선택된 측정 항목 데이터가 없습니다.")

# --- 보조 타임라인 (기존 히트맵 대체 - y 인코딩 추가) ---
st.subheader("Operational Status Timeline (Barge - Detailed)")
if not df_processed_hourly.empty and 'status' in df_processed_hourly.columns and pd.notna(df_processed_hourly['status']).any():
    color_scale_bar = alt.Scale(domain=["Go","Delay","NoGo", "Unknown"], range=["#2ca02c","#ff7f0e","#d62728", "#cccccc"])
    timeline_bar = (alt.Chart(df_processed_hourly).mark_rule(size=6).encode(
            x=alt.X('time:T', title='Time (UTC)', axis=alt.Axis(grid=False)),
            y=alt.value(10),
            color=alt.Color('status:N', scale=color_scale_bar, title='Status', legend=None),
            tooltip=[alt.Tooltip('time:T', title='Time'), 'status:N', 'go_nogo:N', 'risk_reason:N']
        ).properties(height=50))
    st.altair_chart(timeline_bar, use_container_width=True)
else:
    st.info("Operational Status Timeline (Barge)를 표시할 데이터가 없습니다.")

# --- Display Tide Working Windows ---
st.subheader(f"Tide Working Windows (Tide ≥ {TIDE_LIMIT}m, Min 3 consecutive hours)")
if 'tide_m' in df_processed_hourly.columns and 'time' in df_processed_hourly.columns and not df_processed_hourly.empty:
    df_for_windows = df_processed_hourly.copy()
    # Ensure time column is datetime and set as index for find_tide_windows
    df_for_windows['time'] = pd.to_datetime(df_for_windows['time'])
    df_for_windows = df_for_windows.set_index('time', drop=False)
    wins = find_tide_windows(df_for_windows, limit=TIDE_LIMIT, min_hours=3)
    if wins.empty:
        st.info(f"No tide working windows meeting criteria (Tide ≥ {TIDE_LIMIT}m for ≥ 3 hours) found.")
    else:
        st.success("Possible tide working windows:")
        display_wins = wins.copy()
        if 'min' in display_wins.columns:
            display_wins['min'] = pd.to_datetime(display_wins['min']).dt.strftime('%Y-%m-%d %H:%M UTC')
        if 'max' in display_wins.columns:
            display_wins['max'] = (pd.to_datetime(display_wins['max']) + pd.Timedelta(hours=1)).dt.strftime('%Y-%m-%d %H:%M UTC')
        display_wins.rename(columns={'min': 'Window Start', 'max': 'Window End', 'duration_h': 'Duration (Hours)'}, inplace=True)
        st.dataframe(display_wins[['Window Start', 'Window End', 'Duration (Hours)']], use_container_width=True)
else:
    st.info("Tide data ('tide_m') 또는 시간별 예보 데이터에 'time' 컬럼이 없어 작업 가능 창을 계산할 수 없습니다.") 