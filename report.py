# report.py
import argparse, matplotlib.pyplot as plt
import pandas as pd
import os # For TIDE_KEY from environment if not passed explicitly
import datetime as dt # dt 임포트 추가
from fetcher import fetch_and_process_data, assess_operational_risk, LAT, LON, DAYS

# PDF 보고서의 조수 한계선
TIDE_LIMIT_REPORT = 1.5
# PDF 보고서의 파고/풍속 한계선 (Automations Task Prompt 기준)
WAVE_LIMIT_FT_REPORT = 6.0 # ft
WIND_LIMIT_KT_REPORT = 20.0 # knots

def make_pdf_report(outfile: str, selected_date_str: str, tide_api_key: str):
    """주어진 날짜에 대한 예보 데이터를 가져와 PDF 보고서를 생성합니다."""
    
    print(f"Generating PDF report for {selected_date_str} to {outfile}...")
    
    # 사용자의 fetcher.py의 fetch_and_process_data 호출
    # 반환값: df_combined, trend_data, daily_data, error_message
    # fetch_and_process_data가 selected_date_str을 사용하도록 fetcher.py 수정 필요.
    # 현재 fetcher.py는 selected_date_str 인자를 받지 않고, 내부적으로 DAYS (7일) 데이터를 가져옴.
    # report.py는 특정 시작 날짜의 리포트를 생성하므로, fetcher.py가 이를 지원해야 함.
    # 임시로, fetch_and_process_data가 tide_api_key만 받고, 내부적으로 오늘 기준 데이터를 가져온다고 가정.
    # 또는, fetcher.py의 DAYS를 사용. (이 경우 selected_date_str은 제목 표시에만 사용될 수 있음)
    df, _, _, error_message_fetcher = fetch_and_process_data(
        tide_api_key=tide_api_key,
        start_date_str=selected_date_str,
        days_to_fetch=DAYS
    )

    if error_message_fetcher or df.empty:
        error_message_display = error_message_fetcher if error_message_fetcher else "데이터 없음"
        print(f"오류 또는 데이터 부족: {error_message_display}. 날짜: {selected_date_str}")
        fig, ax = plt.subplots(figsize=(8.27, 11.7))
        ax.text(0.5, 0.5, f"PDF 생성 실패: {error_message_display}\n날짜: {selected_date_str}", ha='center', va='center', wrap=True, color='red')
        ax.set_axis_off()
        plt.savefig(outfile)
        print(f"오류 PDF 보고서 저장: {outfile}")
        return False, f"데이터 없음/오류: {error_message_display}"

    # 'time' 컬럼을 datetime으로 변환 (UTC assumed from fetcher)
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC') # Ensure UTC
    else:
        df['time'] = df['time'].dt.tz_convert('UTC')

    # fetcher.py에서 wave_m, wind_kt 컬럼이 이미 생성되어 있다고 가정.
    # tide_m도 포함되어 있다고 가정.
    required_cols = ['time', 'wave_m', 'wind_kt', 'tide_m']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        error_msg = f"필수 컬럼 누락: {missing_cols}"
        print(error_msg)
        # (오류 PDF 생성 로직은 위와 유사하게 추가 가능)
        return False, error_msg

    df['wave_ft'] = df['wave_m'] * 3.28084
    # wind_kt는 이미 fetcher에서 노트 단위로 제공된다고 가정.

    # selected_date_str부터 DAYS 만큼의 데이터 필터링 (만약 fetcher가 더 많이 가져왔을 경우 대비)
    start_date = pd.to_datetime(selected_date_str, format='%Y%m%d', utc=True)
    end_date = start_date + pd.Timedelta(days=DAYS -1) # DAYS일 포함
    df_filtered = df[(df['time'] >= start_date) & (df['time'] <= end_date + pd.Timedelta(days=1, seconds=-1))].copy()
    
    if df_filtered.empty:
        error_msg = f"{selected_date_str}부터 {DAYS}일간 데이터가 없습니다."
        print(error_msg)
        return False, error_msg

    daily_summary = df_filtered.set_index('time').resample('D').agg(
        wave_ft_max=('wave_ft', 'max'),
        wind_kt_max=('wind_kt', 'max'),
        tide_m_min=('tide_m', 'min'),
        tide_m_max=('tide_m', 'max')
    ).reset_index()
    
    daily_summary.dropna(subset=['wave_ft_max', 'wind_kt_max', 'tide_m_min'], how='all', inplace=True)

    if daily_summary.empty:
        error_msg = f"일별 요약 데이터 생성 실패 (날짜: {selected_date_str})"
        print(error_msg)
        return False, error_msg

    # 위험도 평가 적용 (대시보드와 유사하게, 하지만 여기서는 assess_operational_risk 직접 호출)
    if all(col in df.columns for col in ['wave_m', 'wind_kt', 'tide_m']):
        risk_results = df.apply(
            lambda row: assess_operational_risk(
                row.get('wave_m'), row.get('wind_kt'), row.get('tide_m')
            ), axis=1
        )
        risk_df = pd.DataFrame(risk_results.tolist(), index=df.index, columns=['risk_color', 'risk_level', 'go_nogo', 'risk_reason'])
        df = df.join(risk_df)
    else:
        # 필수 컬럼 부족 시 처리 (예: 오류 발생 또는 기본값 할당)
        print("Warning: Could not assess risk for PDF due to missing columns.")
        for col_risk in ['risk_color', 'risk_level', 'go_nogo', 'risk_reason']:
            if col_risk not in df.columns: df[col_risk] = "Unknown"

    # --- ★ 추가: PDF용 시간별 상태 요약 계산 --- 
    # (dashboard의 preprocess 함수 내 로직과 유사하게 구현)
    go_hr_pdf, delay_hr_pdf, nogo_hr_pdf = 0, 0, 0
    if 'go_nogo' in df.columns and 'risk_level' in df.columns:
        df["status_pdf"] = df.apply(
            lambda r: 
                "Go" if r.get('go_nogo') == "GO" 
                else ("Delay" if r.get('risk_level') == "Med" and r.get('go_nogo') != "NO-GO" 
                      else ("NoGo" if r.get('go_nogo') == "NO-GO" else "Unknown")), 
            axis=1
        )
        summary_pdf = df['status_pdf'].value_counts().reindex(["Go","Delay","NoGo"], fill_value=0)
        go_hr_pdf = summary_pdf.get("Go", 0) # 시간별 데이터이므로 count가 시간임
        delay_hr_pdf = summary_pdf.get("Delay", 0)
        nogo_hr_pdf = summary_pdf.get("NoGo", 0)
        
    pdf_summary_badge = f"[Summary: Go {go_hr_pdf}h | Delay {delay_hr_pdf}h | NoGo {nogo_hr_pdf}h]"
    print(f"PDF Summary Badge: {pdf_summary_badge}")

    overall_recommendation = "✅ Sailing OK."
    recommendation_reasons = []
    for _, row in daily_summary.iterrows():
        if pd.notna(row['tide_m_min']) and row['tide_m_min'] < TIDE_LIMIT_REPORT:
            overall_recommendation = "⚠️ Barge sailing NOT recommended"
            recommendation_reasons.append(f"{row['time'].strftime('%m-%d')}: Low tide ({row['tide_m_min']:.2f}m)")
        if pd.notna(row['wave_ft_max']) and row['wave_ft_max'] > WAVE_LIMIT_FT_REPORT:
            overall_recommendation = "⚠️ Barge sailing NOT recommended"
            recommendation_reasons.append(f"{row['time'].strftime('%m-%d')}: High wave ({row['wave_ft_max']:.1f}ft)")
        if pd.notna(row['wind_kt_max']) and row['wind_kt_max'] > WIND_LIMIT_KT_REPORT:
            overall_recommendation = "⚠️ Barge sailing NOT recommended"
            recommendation_reasons.append(f"{row['time'].strftime('%m-%d')}: High wind ({row['wind_kt_max']:.1f}kt)")
    
    report_summary_message = overall_recommendation
    if recommendation_reasons:
        report_summary_message += " Reasons: " + "; ".join(list(set(recommendation_reasons)))
    print(f"Overall Recommendation for report: {report_summary_message}")

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.7 * 0.9), sharex=True) # figsize 약간 줄임
    report_title_date = pd.to_datetime(selected_date_str, format='%Y%m%d').strftime('%Y-%m-%d')
    # 제목에 report_summary_message와 pdf_summary_badge 모두 포함
    full_report_title = (f"{DAYS}-Day Marine Forecast for Al Ghallan (from {report_title_date})\n"
                         f"{report_summary_message}\n{pdf_summary_badge}")
    fig.suptitle(full_report_title, fontsize=11, y=0.99) # 폰트 크기 약간 줄임

    dates_for_plot = daily_summary['time'].dt.strftime('%m-%d (%a)')

    axes[0].plot(dates_for_plot, daily_summary['wave_ft_max'], 's-', label='Max Wave Height (ft)', color='#1f77b4')
    axes[0].axhline(WAVE_LIMIT_FT_REPORT, color='r', ls='--', label=f'Limit ({WAVE_LIMIT_FT_REPORT} ft)')
    axes[0].set_ylabel('Wave Height (ft)')
    axes[0].legend(fontsize='small')
    axes[0].grid(True, linestyle=':')

    axes[1].plot(dates_for_plot, daily_summary['wind_kt_max'], 'o-', label='Max Wind Speed (knots)', color='#ff7f0e')
    axes[1].axhline(WIND_LIMIT_KT_REPORT, color='r', ls='--', label=f'Limit ({WIND_LIMIT_KT_REPORT} knots)')
    axes[1].set_ylabel('Wind Speed (knots)')
    axes[1].legend(fontsize='small')
    axes[1].grid(True, linestyle=':')

    axes[2].plot(dates_for_plot, daily_summary['tide_m_min'], '^-', label='Min Tide Level (m)', color='#2ca02c')
    if 'tide_m_max' in daily_summary.columns and pd.notna(daily_summary['tide_m_max']).any():
        axes[2].plot(dates_for_plot, daily_summary['tide_m_max'], 'v-', label='Max Tide Level (m)', color='#2ca02c', alpha=0.5)
    axes[2].axhline(TIDE_LIMIT_REPORT, color='r', ls='--', label=f'Min Limit ({TIDE_LIMIT_REPORT} m)')
    axes[2].set_ylabel('Tide Level (m)')
    axes[2].legend(fontsize='small')
    axes[2].grid(True, linestyle=':')
    
    plt.xlabel("Date")
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.94]) # 제목 길어졌으므로 rect 조정
    
    try:
        plt.savefig(outfile)
        print(f"PDF report successfully saved to {outfile}")
        return True, report_summary_message
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return False, f"PDF 저장 오류: {e}"
    finally:
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a PDF marine forecast report.")
    parser.add_argument('--pdf', required=True, help='Output PDF file path')
    parser.add_argument('--date', required=False, help='Start date for the forecast (YYYYMMDD). Defaults to today.',
                        default=dt.datetime.now().strftime('%Y%m%d')) # dt 사용
    
    # TIDE_KEY를 환경변수에서 가져오도록 처리. 명령행 인자로도 받을 수 있게 확장 가능.
    # 스크립트 실행 환경에 따라 st.secrets 사용이 불가할 수 있으므로, 환경변수를 우선 고려.
    tide_api_key_env = os.getenv("TIDE_KEY")
    if not tide_api_key_env:
        print("Error: TIDE_KEY environment variable not set. This is required to fetch tide data.")
        # 개발/테스트 목적으로 하드코딩된 키를 사용하거나, 여기서 실행을 중단할 수 있습니다.
        # 예: raise ValueError("TIDE_KEY environment variable must be set.")
        # 여기서는 경고만 하고 진행 (fetcher에서 API키 없으면 빈 데이터 반환)

    args = parser.parse_args()
    
    success, message = make_pdf_report(args.pdf, args.date, tide_api_key_env)
    
    if success:
        print(f"Report generation successful. Summary: {message}")
    else:
        print(f"Report generation failed. Reason: {message}")

# Add a placeholder for fetcher.py if it's not available for context
# This part is for completeness if fetcher.py is missing.
# If fetcher.py is present, this won't be used.
try:
    from fetcher import fetch_and_process_data, LAT, LON
except ImportError:
    print("Warning: 'fetcher.py' not found. Using placeholder data for 'fetch_and_process_data'.")
    print("Please ensure 'fetcher.py' is in the same directory or accessible in PYTHONPATH.")
    
    # Define LAT, LON if not imported (example values)
    LAT = 0.0 
    LON = 0.0

    def fetch_and_process_data():
        """Placeholder function if fetcher.py is not found."""
        # Create a dummy DataFrame that matches the expected structure
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create a date range for 7 days
        base_date = datetime.now()
        date_list = [base_date + timedelta(days=i, hours=h) for i in range(7) for h in range(0, 24, 3)]
        
        data = {
            'time': date_list,
            # wave_m: random values between 0.1 and 3 meters
            'wave_m': np.random.uniform(0.1, 3.0, size=len(date_list)),
            # wind_kt: random values between 1 and 30 knots
            'wind_kt': np.random.uniform(1.0, 30.0, size=len(date_list))
        }
        df = pd.DataFrame(data)
        # Ensure 'time' is datetime
        df['time'] = pd.to_datetime(df['time'])
        return df, "dummy_source.csv", {} # Return df and other expected values 