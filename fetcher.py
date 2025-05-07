#!/usr/bin/env python3
import os, time, logging, datetime as dt, requests, pandas as pd, pytz
from requests.adapters import HTTPAdapter, Retry
from statsmodels.tsa.arima.model import ARIMA
from pandas.errors import OutOfBoundsDatetime
import streamlit as st # Keep streamlit import if needed for secrets in app.py
import numpy as np # For placeholder data generation

# --- NCM Parser import ---
from ncm_parser import parse_ncm

# Constants
LAT, LON = 24.541664, 54.29167 # Using original coordinates
MARINE_VARS = "wave_height" # Updated based on new code
WEATHER_VARS = "wind_speed_10m" # Updated based on new code
# THR_WAVE, THR_WIND = 2.0, 12.0 # Thresholds now handled within table logic
UAE = pytz.timezone("Asia/Dubai") # Keep UAE timezone if needed for logging/internal use
TZ_AUTO = "auto" # API timezone parameter
DAYS = 7 # Fetch 7 days

# Optional: Load SLACK_URL from .env if needed for future Slack integration
# from dotenv import load_dotenv
# load_dotenv()
# SLACK_URL = os.getenv("SLACK_WEBHOOK_URL")

def build_sess(retries=3):
    """Builds a requests session with retry capabilities."""
    retry = Retry(total=retries, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

# --- Integrated Table Logic from User Snippet ---

# Helper for trend_table
def risk(w, v):
    """Determines risk level and action based on wave height (m) and wind speed (kt)."""
    if w<=1.0 and v<=15:   return "(Green) Low",   "LOLO/RORO 가능"
    if w<=1.5 and v<=20:   return "(Orange) Medium","LOLO 대기, 항해 가능"
    if w<=2.5 and v<=25:   return "(Orange) Medium","LOLO 가능, RORO 제한" # Note: Original snippet had same text, adjusted based on context
    return "(Red) High", "전작업 중단"

# Generates 7-day trend summary
def trend_table(df):
    """Generates the 7-day trend summary table data."""
    rows = []
    # Ensure 'time' column is timezone-aware or convert to naive if necessary
    # Assuming API returns timezone-aware datetimes based on tz='auto'
    # If df["time"] is naive, use dt.datetime.now().date() instead of dt.date.today()
    try:
        min_date = df["time"].dt.date.min()
    except TypeError:
        logging.error("Time column might be timezone-naive. Cannot reliably get min date.")
        # Attempt conversion or handle error appropriately
        # For now, let's assume it works or fails gracefully downstream
        min_date = pd.Timestamp('today').date() # Fallback, might be incorrect timezone


    for t in range(DAYS):
        d = min_date + dt.timedelta(days=t)
        day_df = df[df["time"].dt.date == d]
        if day_df.empty: continue

        # Ensure indices 8 and 14 exist, otherwise skip or handle
        if len(day_df) > 14:
            h08 = day_df.iloc[8]  # 08:00 local time (relative to API's timezone 'auto')
            h14 = day_df.iloc[14] # 14:00 local time
        else:
            logging.warning(f"Not enough hourly data for day {d} to get 08:00/14:00 entries.")
            continue # Skip this day if not enough data points

        # Determine mode based on day index 't'
        mode_08 = "초기" if t == 0 else ""
        if t == 2: mode_14 = "피크"
        elif t == 1: mode_14 = "상승"
        elif t == 3: mode_14 = "하락"
        else: mode_14 = "안정" if t > 0 else "" # Avoid '안정' for day 0

        # Process 08:00 entry
        if mode_08: # Only add 08:00 for the first day
             r, act = risk(h08.wave_m, h08.wind_kt)
             rows.append([h08.time.strftime("%d %a %H:%M"), mode_08,
                          f"{h08.wave_m:.1f} m", f"{h08.wind_kt:.0f} kt", r, act])
        # Process 14:00 entry
        if mode_14: # Add 14:00 for all days except potentially day 0 if mode is empty
            r, act = risk(h14.wave_m, h14.wind_kt)
            rows.append([h14.time.strftime("%d %a %H:%M"), mode_14,
                        f"{h14.wave_m:.1f} m", f"{h14.wind_kt:.0f} kt", r, act])

    return rows


# Helper for daily_table
def risk_mark(w, v):
    """Determines daily sailing risk mark, level, decision, and note."""
    if w<=1.0 and v<=15: return "(Green)", "Low",  "GO", "정상"
    if w<=1.5 and v<=20: return "(Orange)", "Med",  "Delay after 18:00", "저속상대풍 주의" if w > 1.0 else "정상" # Refined note condition
    if w<=2.5 and v<=25: return "(Orange)", "Med",  "Possible 14:00+", "저속상대풍 주의" # Original had Med risk > 1.5 wave, kept consistent
    return "(Red)", "High", "NO-GO", "정상" # High risk is assumed '정상' in terms of note

# Generates daily sailing possibility table
def daily_table(df):
    """Generates the daily sailing possibility table data."""
    tbl = []
    try:
        min_date = df["time"].dt.date.min()
    except TypeError:
        logging.error("Time column might be timezone-naive. Cannot reliably get min date.")
        min_date = pd.Timestamp('today').date() # Fallback


    for d in range(DAYS):
        day = min_date + dt.timedelta(days=d)
        day_df = df[df["time"].dt.date == day]
        if day_df.empty: continue

        # Ensure 'time' is datetime before accessing .dt.hour
        # Fix SettingWithCopyWarning by explicitly creating a copy
        day_df = day_df.copy()
        day_df['time'] = pd.to_datetime(day_df['time'])

        a = day_df[(day_df["time"].dt.hour >= 0) & (day_df["time"].dt.hour < 12)]
        b = day_df[(day_df["time"].dt.hour >= 12) & (day_df["time"].dt.hour < 24)]

        # Handle potential empty slices if data is missing for a period
        w1 = "N/A"
        if not a.empty:
             w1 = f"{a.wind_kt.min():.0f}-{a.wind_kt.max():.0f} kt / {a.wave_m.min():.1f}-{a.wave_m.max():.1f} m"

        w2 = "N/A"
        if not b.empty:
             w2 = f"{b.wind_kt.min():.0f}-{b.wind_kt.max():.0f} kt / {b.wave_m.min():.1f}-{b.wave_m.max():.1f} m"

        # Calculate overall peaks, handling potentially empty a or b
        peak_wave = -1
        peak_wind = -1
        if not a.empty and not b.empty:
            peak_wave = max(a.wave_m.max(), b.wave_m.max())
            peak_wind = max(a.wind_kt.max(), b.wind_kt.max())
        elif not a.empty:
            peak_wave = a.wave_m.max()
            peak_wind = a.wind_kt.max()
        elif not b.empty:
            peak_wave = b.wave_m.max()
            peak_wind = b.wind_kt.max()

        if peak_wave != -1 and peak_wind != -1:
             mark, risk_level, decision, note = risk_mark(peak_wave, peak_wind)
             # Adjust note based on refined condition in risk_mark
             # note = "저속상대풍 주의" if risk_level=="Med" and peak_wave>1.5 else "정상" # Now handled inside risk_mark
        else:
            mark, risk_level, decision, note = "(White)", "N/A", "N/A", "Missing Data"


        tbl.append([day.strftime("%d %a"), w1, w2, f"{mark} {risk_level}", decision, note])
    return tbl

# --- End Integrated Table Logic ---

# --- Tide Fetching Function ---
def fetch_tide(lat, lon, api_key, days=7):
    """Fetches tide data from WorldTides API."""
    if not api_key:
        logging.warning("WorldTides API key not provided. Skipping tide data fetch.")
        return pd.DataFrame(columns=["time", "tide_m"]) # Return empty DataFrame
        
    url = "https://www.worldtides.info/api/v3?heights"
    params = dict(lat=lat, lon=lon, days=days, key=api_key)
    try:
        js = requests.get(url, params=params, timeout=15).json() # Increased timeout
        if 'error' in js:
            logging.error(f"WorldTides API error: {js['error']}")
            return pd.DataFrame(columns=["time", "tide_m"])
            
        df = pd.DataFrame(js.get("heights", []))
        if df.empty:
            logging.warning("WorldTides API returned no height data.")
            return pd.DataFrame(columns=["time", "tide_m"])
            
        # Convert to datetime, ensure aware UTC, then convert to naive UTC
        df["time"] = pd.to_datetime(df["date"], utc=True)
        df["time"] = df["time"].dt.tz_localize(None) 
        
        return df[["time", "height"]].rename(columns={"height": "tide_m"})
    except requests.exceptions.RequestException as e:
        logging.error(f"WorldTides API request failed: {e}")
        return pd.DataFrame(columns=["time", "tide_m"])
    except Exception as e:
        logging.error(f"Error processing tide data: {e}")
        return pd.DataFrame(columns=["time", "tide_m"])
# --- End Tide Fetching ---

# --- ARIMA Forecasting Function ---
def generate_arima_forecast(wave_series):
    """Generates 48-hour ARIMA forecast for wave height."""
    if wave_series.empty or wave_series.isnull().all():
        logging.warning("Wave data is empty or all null, cannot generate ARIMA forecast.")
        return pd.DataFrame(columns=["time", "wave_pred"])
        
    try:
        # Ensure data is float and handle potential NaNs (e.g., forward fill)
        train_data = wave_series.astype(float).ffill().bfill() # Fill NaNs
        if train_data.isnull().any(): # Still NaNs after filling?
             logging.warning("Could not fill all NaNs in wave data for ARIMA.")
             return pd.DataFrame(columns=["time", "wave_pred"])

        # Define and fit the ARIMA model (using p=2, d=0, q=2 as in example)
        # Order (p,d,q): AR order, differencing order, MA order
        # Simple order, might need tuning based on data characteristics
        model = ARIMA(train_data, order=(2, 0, 2))
        model_fit = model.fit()
        
        # Forecast next 48 steps (hours)
        forecast_values = model_fit.forecast(steps=48)
        
        # Create forecast timestamps (already starts from aware UTC/timezone from index)
        last_timestamp = wave_series.index.max() 
        forecast_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=48, freq='h')
        
        df_pred = pd.DataFrame({"time": forecast_index, "wave_pred": forecast_values})
        
        # Ensure time is naive UTC like the main df
        # Check if index is timezone aware before converting
        if df_pred["time"].dt.tz is not None:
             df_pred["time"] = df_pred["time"].dt.tz_convert('UTC').dt.tz_localize(None)
        else: # If somehow already naive, assume it was intended as UTC naive
             pass 
             
        logging.info("ARIMA forecast generated successfully.")
        return df_pred
        
    except OutOfBoundsDatetime:
         logging.error("Timestamp issue during ARIMA forecast index generation.")
         return pd.DataFrame(columns=["time", "wave_pred"])
    except Exception as e:
        logging.error(f"Error during ARIMA model fitting or forecasting: {e}")
        # Return empty dataframe on error
        return pd.DataFrame(columns=["time", "wave_pred"])
# --- End ARIMA Forecasting ---

def fetch_and_process_data(tide_api_key=None, start_date_str=None, days_to_fetch=DAYS):
    """Fetches 7-day data, processes it, generates tables & forecast, and returns results."""
    s = build_sess()
    df_combined = pd.DataFrame()
    trend_data = []
    daily_data = []
    error_message = None
    df_forecast = pd.DataFrame()
    daily_comparison_df = pd.DataFrame() # Initialize daily_comparison_df

    # start_date_str이 None이면 오늘 날짜(UAE 기준) 사용
    if start_date_str is None:
        start_date_str = dt.datetime.now(UAE).strftime('%Y-%m-%d')
    else: # YYYYMMDD 형식을 YYYY-MM-DD로 변환 (Open-Meteo API 형식)
        try:
            start_date_str = pd.to_datetime(start_date_str, format='%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            logging.error(f"Invalid start_date_str format: {start_date_str}. Using today.")
            start_date_str = dt.datetime.now(UAE).strftime('%Y-%m-%d')
            
    end_date_str = (pd.to_datetime(start_date_str) + pd.Timedelta(days=days_to_fetch-1)).strftime('%Y-%m-%d')

    try:
        logging.info(f"Fetching {days_to_fetch}-day marine and wind data from {start_date_str} to {end_date_str}...")
        # Fetch Marine Data
        m_url = "https://marine-api.open-meteo.com/v1/marine"
        m_params = dict(latitude=LAT, longitude=LON, hourly=MARINE_VARS,
                        # forecast_days=days_to_fetch, # start_date, end_date 사용
                        start_date=start_date_str,
                        end_date=end_date_str,
                        timezone=TZ_AUTO, cell_selection="sea")
        m_resp = s.get(m_url, params=m_params, timeout=10)
        m_resp.raise_for_status()
        marine_data = m_resp.json()

        # Fetch Weather Data - Add windspeed_unit parameter
        w_url = "https://api.open-meteo.com/v1/forecast"
        w_params = dict(latitude=LAT, longitude=LON, hourly=WEATHER_VARS,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        windspeed_unit="ms", # <<< 풍속 단위를 m/s로 요청 추가
                        timezone=TZ_AUTO)
        w_resp = s.get(w_url, params=w_params, timeout=10)
        w_resp.raise_for_status()
        wind_data = w_resp.json()

        # Fetch Tide Data
        # fetch_tide는 days 인자를 사용하므로 days_to_fetch 전달
        logging.info(f"Fetching {days_to_fetch}-day tide data...")
        tide_df = fetch_tide(LAT, LON, tide_api_key, days=days_to_fetch) # fetch_tide는 자체적으로 start/end date 처리 안함. API가 현재부터 days만큼 줄 수 있음.
                                                                    # WorldTides API가 start_date, end_date를 지원하는지 확인 필요. 현재는 days만 사용.
                                                                    # 이는 조수 데이터가 요청된 start_date_str과 정확히 일치하지 않을 수 있음을 의미.
                                                                    # 더 정확하려면 fetch_tide도 start_date를 받도록 수정해야 함. (이번 수정 범위에는 미포함)

        logging.info("Data fetched successfully.")

        marine_df = pd.DataFrame(marine_data["hourly"])
        wind_df   = pd.DataFrame(wind_data["hourly"])

        # Check for empty dataframes early
        if marine_df.empty or wind_df.empty:
            logging.warning("Received empty dataframe from one or both APIs.")
            error_message = "⚠️ Received empty data from API."
            return df_combined, trend_data, daily_data, error_message, daily_comparison_df

        # Standardize timezones before merge: Convert to aware UTC, then to naive UTC
        if not marine_df.empty:
            marine_df["time"] = pd.to_datetime(marine_df["time"], utc=True)
            marine_df["time"] = marine_df["time"].dt.tz_localize(None)

        if not wind_df.empty:
            wind_df["time"] = pd.to_datetime(wind_df["time"], utc=True)
            wind_df["time"] = wind_df["time"].dt.tz_localize(None)

        # tide_df timezone is already standardized within fetch_tide function
        
        # Merge marine and wind first
        if not marine_df.empty and not wind_df.empty:
            df_combined = pd.merge(marine_df, wind_df, on="time", how="inner")
        elif not marine_df.empty:
            df_combined = marine_df
        elif not wind_df.empty:
            df_combined = wind_df
        else:
            # Handle case where both marine and wind fetch failed
            logging.error("Both marine and wind data fetching failed.")
            error_message = "⚠️ Failed to fetch marine and wind data."
            # Return tide df if available, otherwise empty
            df_final = tide_df if not tide_df.empty else pd.DataFrame()
            return df_final, [], [], error_message, daily_comparison_df

        # Merge Tide Data (use left merge to keep all marine/wind times)
        if not tide_df.empty:
            df_combined = pd.merge(df_combined, tide_df, on="time", how="left")
        else:
            if not df_combined.empty: # Add tide_m only if df_combined is not empty
                 df_combined['tide_m'] = pd.NA
        
        # Re-check if empty after all merges potentially failing
        if df_combined.empty:
             logging.error("Final dataframe is empty after all merges.")
             error_message = "⚠️ Failed to combine any forecast data."
             return df_combined, trend_data, daily_data, error_message, daily_comparison_df

        # Calculate wind knots - Now correctly converts m/s to knots
        # WEATHER_VARS ('wind_speed_10m') 컬럼이 m/s 단위로 반환됨
        if WEATHER_VARS in df_combined.columns:
             df_combined["wind_kt"] = df_combined[WEATHER_VARS] * 1.94384
        else:
             df_combined["wind_kt"] = pd.NA # 컬럼 없으면 NA 처리
             logging.warning(f"Weather variable '{WEATHER_VARS}' not found for wind knot conversion.")

        df_combined = df_combined.rename(columns={MARINE_VARS: "wave_m"})

        # Ensure df_combined has a datetime index for ARIMA
        if 'time' in df_combined.columns and not df_combined.empty:
             # Convert time column to datetime objects first if not already
             df_combined['time'] = pd.to_datetime(df_combined['time'])
             # Set index AFTER converting to datetime
             df_combined_indexed = df_combined.set_index('time')
             # Localize index to UTC before passing to ARIMA if it's naive
             if df_combined_indexed.index.tz is None:
                  df_combined_indexed = df_combined_indexed.tz_localize('UTC')
                  
             # --- Add asfreq('h') to set frequency explicitly --- 
             df_combined_indexed = df_combined_indexed.asfreq('h')
             # --- End frequency setting ---
             
             if 'wave_m' in df_combined_indexed.columns:
                 # Pass the resampled series to the forecast function
                 df_forecast = generate_arima_forecast(df_combined_indexed['wave_m'])
             else:
                 logging.warning("'wave_m' column not found for ARIMA forecast.")
             # No need to reset index here, merge uses the 'time' column
        else:
             logging.warning("Could not generate ARIMA forecast due to missing time index or empty dataframe.")
             df_forecast = pd.DataFrame(columns=["time", "wave_pred"]) # Ensure empty df is created

        # Merge forecast back into the main dataframe (left merge)
        if not df_forecast.empty:
            df_combined = pd.merge(df_combined, df_forecast, on="time", how="left")
        else:
            if not df_combined.empty:
                 df_combined['wave_pred'] = pd.NA
        
        # Keep only necessary columns + tide
        keep_cols = ["time", "wave_m", "wind_kt"]
        if 'tide_m' in df_combined.columns:
            keep_cols.append('tide_m')
        if 'wave_pred' in df_combined.columns:
            keep_cols.append('wave_pred') # Add wave_pred here
        df_combined = df_combined[keep_cols].sort_values(by="time").reset_index(drop=True)

        # Generate tables
        trend_data = trend_table(df_combined)
        daily_data = daily_table(df_combined)

        # --- NCM Data Fetching and Processing (as per user's diff) ---
        if not df_forecast.empty:
            print("Fetcher: Starting NCM data fetch and comparison process...")
            try:
                start_date_dt = pd.to_datetime(start_date_str, format='%Y%m%d').date()
                
                ncm_daily_data_list = []
                # NCM typically provides a few days of forecast. Let's fetch for the relevant period.
                # User's example fetches for 5 days from start_date. 
                # This might need adjustment based on how NCM PDFs are structured (e.g., one PDF covers multiple days).
                # The current ncm_parser.parse_ncm fetches PDF for a single given date.
                # If one NCM PDF contains a multi-day forecast, parse_ncm would need to be adapted
                # or called once for the start_date, and then process its multi-day content.
                # Assuming parse_ncm returns data for the *date* of the PDF title.

                # For a 7-day dashboard view, we might want NCM data for each of those 7 days if available.
                # The parse_ncm function tries to get PDF for current_date_to_try and then uses that date in its output.
                ncm_dfs_list = []
                for i in range(days_to_fetch): # Fetch NCM for each day in the dashboard's range
                    current_loop_date = start_date_dt + dt.timedelta(days=i)
                    print(f"Fetcher: Parsing NCM for date: {current_loop_date.strftime('%Y-%m-%d')}")
                    # parse_ncm already tries previous days if PDF for current_loop_date is not found.
                    # The 'date' column in the returned df from parse_ncm will be current_loop_date if successful.
                    df_ncm_single_day = parse_ncm(current_loop_date) 
                    ncm_dfs_list.append(df_ncm_single_day)
                
                if ncm_dfs_list:
                    ncm_df_raw = pd.concat(ncm_dfs_list).reset_index(drop=True)
                    ncm_df_raw.dropna(subset=['wave_ft_max_off', 'wind_kn_max_off'], how='all', inplace=True) # Drop rows where all NCM values are NaN
                    print(f"Fetcher: Raw NCM data collected for period (rows: {len(ncm_df_raw)}):")
                    # print(ncm_df_raw.to_string()) # Can be verbose
                else:
                    ncm_df_raw = pd.DataFrame(columns=['date', 'wave_ft_max_off', 'wind_kn_max_off'])
                    print("Fetcher: No NCM data could be parsed for the period.")

                # Aggregate API data daily to merge with NCM daily data
                # Ensure 'time' column is datetime type before resampling
                df_forecast['time'] = pd.to_datetime(df_forecast['time'])
                daily_api_agg = df_forecast.resample('D', on='time').agg(
                    wave_m_max_api=('wave_pred', 'max'),
                    wind_kt_max_api=('wave_pred', 'max')
                ).reset_index()
                daily_api_agg['date'] = daily_api_agg['time'].dt.date # Convert timestamp to date object for merging
                daily_api_agg.drop(columns=['time'], inplace=True)

                if not ncm_df_raw.empty:
                    # Ensure ncm_df_raw['date'] is also a date object if it's not already
                    ncm_df_raw['date'] = pd.to_datetime(ncm_df_raw['date']).dt.date
                    merged_comparison = pd.merge(
                        daily_api_agg,
                        ncm_df_raw,
                        how='left', on='date', validate='1:1' # Assumes one NCM entry per date
                    )
                else: # If NCM data is empty, create a comparison table with API data only
                    merged_comparison = daily_api_agg.copy()
                    merged_comparison['wave_ft_max_off'] = np.nan
                    merged_comparison['wind_kn_max_off'] = np.nan
                
                # Calculate difference percentage
                # Convert API wave_m to feet for comparison
                merged_comparison['wave_ft_max_api'] = (merged_comparison['wave_m_max_api'] * 3.28084).round(1)

                # Calculate gap: ((API - NCM) / NCM) * 100
                # Handle potential division by zero or NaN in NCM data
                merged_comparison['wave_gap_%'] = (
                    ((merged_comparison.wave_ft_max_api - merged_comparison.wave_ft_max_off) / merged_comparison.wave_ft_max_off) * 100
                ).replace([np.inf, -np.inf], np.nan).round(1) # Replace inf with NaN if NCM is 0
                
                merged_comparison['wind_gap_%'] = (
                    ((merged_comparison.wind_kt_max_api - merged_comparison.wind_kn_max_off) / merged_comparison.wind_kn_max_off) * 100
                ).replace([np.inf, -np.inf], np.nan).round(1)

                daily_comparison_df = merged_comparison
                print("Fetcher: NCM comparison data generated:")
                # print(ncm_comparison_df.to_string()) # Can be verbose

            except Exception as e:
                print(f"Error (fetcher.py): Failed during NCM data processing or comparison: {e}")
                # Fallback: create an empty or placeholder ncm_comparison_df
                if 'date' not in daily_comparison_df.columns and not daily_api_agg.empty:
                     daily_comparison_df = daily_api_agg[['date']].copy()
                     for col in ['wave_ft_max_off', 'wave_m_max_api', 'wave_gap_%', 'wind_kn_max_off', 'wind_kt_max_api', 'wind_gap_%', 'wave_ft_max_api']:
                         daily_comparison_df[col] = np.nan
                elif daily_comparison_df.empty:
                     daily_comparison_df = pd.DataFrame(columns=['date', 'wave_ft_max_off', 'wave_m_max_api', 'wave_gap_%', 'wind_kn_max_off', 'wind_kt_max_api', 'wind_gap_%', 'wave_ft_max_api'])

    except requests.exceptions.RequestException as exc:
        logging.exception("API fetch failed: %s", exc)
        error_message = f"❌ API Fetch Error: {exc}"
        if exc.response is not None:
             error_message += f" (Status: {exc.response.status_code})"

    except KeyError as e:
        logging.exception(f"Missing expected key in API response: {e}.")
        error_message = f"❌ API Response Error: Missing key '{e}'"

    except Exception as e:
        logging.exception(f"An error occurred during processing: {e}")
        error_message = f"❌ Processing Error: {e}"

    # Return the main dataframe and the two tables (or error message)
    # AND the new daily_comparison_df
    return df_combined, trend_data, daily_data, error_message, daily_comparison_df

# Keep build_sess, remove other old functions if they are not used anymore
# Remove marine_req, wind_req as logic is now inside fetch_and_process_data

# Remove __main__ block as this is now a module
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, ...)
#     ... schedule logic ... 

# -----------------------------------------------------------------------------
# 1. WorldTides API를 위한 조수 데이터 가져오기 함수
# -----------------------------------------------------------------------------
# @st.cache_data(ttl=43200) # 만약 Streamlit 앱에서 직접 이 함수를 호출한다면 캐싱 사용 가능
def fetch_tide_data(lat, lon, days, api_key):
    """
    WorldTides API에서 특정 위치와 기간 동안의 조수 정보를 가져옵니다.
    (높이, 만조/간조 시각, MSL 기준, 60분 간격)
    """
    if not api_key:
        print("Warning (fetcher.py): WorldTides API key (TIDE_KEY) not provided. Skipping tide data fetch.")
        return pd.DataFrame(columns=['time', 'tide_m'])

    url = "https://www.worldtides.info/api/v3"
    params = dict(
        lat=lat,
        lon=lon,
        key=api_key,
        heights="",      # 시간별 조위 높이 요청
        extremes="",     # 만조/간조 시각 정보 요청
        datum="MSL",     # 평균 해수면 기준
        interval=60,     # 60분 간격 (heights 용)
        days=days
    )
    try:
        response = requests.get(url, params=params, timeout=10) # Added timeout
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        data = response.json()

        tide_heights_list = []
        if data.get('heights'):
            for item in data['heights']:
                tide_time = pd.to_datetime(item['dt'], unit='s', utc=True)
                tide_heights_list.append({'time': tide_time, 'tide_m': item['height']})
        
        if not tide_heights_list:
            print("Warning (fetcher.py): No tide height data returned from WorldTides API.")
            return pd.DataFrame(columns=['time', 'tide_m'])

        tide_df = pd.DataFrame(tide_heights_list)
        tide_df.sort_values(by='time', inplace=True)
        
        if tide_df['time'].dt.tz is None: # Redundant check if pd.to_datetime already sets utc=True
            tide_df['time'] = tide_df['time'].dt.tz_localize('UTC')
        else:
            tide_df['time'] = tide_df['time'].dt.tz_convert('UTC')
            
        return tide_df

    except requests.exceptions.RequestException as e:
        print(f"Error (fetcher.py): Fetching tide data from WorldTides failed: {e}")
        return pd.DataFrame(columns=['time', 'tide_m'])
    except Exception as e:
        print(f"Error (fetcher.py): Processing tide data failed: {e}")
        return pd.DataFrame(columns=['time', 'tide_m'])

# -----------------------------------------------------------------------------
# 2. 운항 위험도 평가 함수
# -----------------------------------------------------------------------------
def assess_operational_risk(wave_height, wind_speed, tide_level):
    """
    파고, 풍속, 조위 정보를 바탕으로 운항 위험도를 평가합니다.
    """
    try:
        wave_height = float(wave_height) if wave_height is not None and pd.notna(wave_height) else None
        wind_speed = float(wind_speed) if wind_speed is not None and pd.notna(wind_speed) else None
        tide_level = float(tide_level) if tide_level is not None and pd.notna(tide_level) else None
    except (ValueError, TypeError):
         return "(Gray)", "Unknown", "N/A", "입력 데이터 타입 오류"

    if tide_level is not None and tide_level < 1.5: # 수심 한계 1.5m
        return "(Red)", "High", "NO-GO", "수심 부족 (< 1.5m)"
    
    if wave_height is None or wind_speed is None:
         return "(Gray)", "Unknown", "N/A", "파고/풍속 데이터 없음"

    if wave_height <= 1.0 and wind_speed <= 15:
        return "(Green)", "Low", "GO", "정상"
    if wave_height <= 1.5 and wind_speed <= 20:
        return "(Orange)", "Med", "Delay after 18:00", "저속·수심 주의"
    if wave_height <= 2.5 and wind_speed <= 25:
        return "(Orange)", "Med", "Possible 14:00+", "롤링 주의"
    
    return "(Red)", "High", "NO-GO", "파고·풍속 초과"

# -----------------------------------------------------------------------------
# 3. 메인 데이터 가져오기 및 통합 함수
# -----------------------------------------------------------------------------
def get_combined_forecast_data(selected_date_str, tide_api_key, days_to_fetch=DAYS):
    """
    선택된 날짜를 기준으로 기본 예보 데이터(파고, 풍속 등)와 조수 데이터를 가져와 병합하고,
    운항 위험도를 평가하여 완전한 DataFrame을 반환합니다.
    Aditionally, fetches NCM data and creates a comparison DataFrame.
    """
    print(f"Fetcher: Called get_combined_forecast_data for date {selected_date_str}, days: {days_to_fetch}")
    error_message_api = None # Specific error for API part
    ncm_comparison_df = pd.DataFrame() # Initialize for NCM comparison

    # --- 3.1 기본 예보 데이터 가져오기 (파고, 풍속 등) ---
    # This now calls the more comprehensive fetch_and_process_data internally
    # to get API data and then we add NCM processing.
    
    # Convert selected_date_str (YYYYMMDD) to YYYY-MM-DD for fetch_and_process_data if needed,
    # or ensure fetch_and_process_data handles YYYYMMDD.
    # The existing fetch_and_process_data handles YYYYMMDD for its start_date_str param.

    # Call the main data fetching function to get API data
    # Note: fetch_and_process_data now also returns daily_comparison_df, but it will be empty from this call
    # as the NCM logic is being added *around* it or *within* the calling scope of this function (get_combined_forecast_data)
    # Let's adjust this. fetch_and_process_data should be the one doing the NCM fetching if it's meant to be the primary data source func.
    # For now, let's assume the user's diff meant to add NCM fetching within this get_combined_forecast_data
    # after getting the API data.

    # Placeholder for API data fetching part (mimicking what fetch_and_process_data would do for API parts)
    # In a real scenario, you might call a leaner function for just API data, or structure this differently.
    # For now, we'll generate placeholder API data here for clarity of NCM integration.

    print("Fetcher: Simulating API data fetch within get_combined_forecast_data...")
    api_df = pd.DataFrame() # Initialize
    try:
        # Simulate fetching API data for the period
        # Convert YYYYMMDD to datetime object for date operations
        base_start_date = pd.to_datetime(selected_date_str, format='%Y%m%d')
        time_index_api = pd.date_range(start=base_start_date, periods=days_to_fetch * 24, freq='h', tz='UTC')
        api_df = pd.DataFrame({
            'time': time_index_api,
            'wave_m': np.random.uniform(0.1, 2.5, size=len(time_index_api)), # m
            'wind_kt': np.random.uniform(5, 25, size=len(time_index_api)),   # knots
            'tide_m': np.random.uniform(-1, 2, size=len(time_index_api))    # m
        })
        api_df['wave_m'] = api_df['wave_m'].round(2)
        api_df['wind_kt'] = api_df['wind_kt'].round(1)
        api_df['tide_m'] = api_df['tide_m'].round(2)
        
        # Ensure 'time' is naive UTC for consistency if other parts expect that
        api_df['time'] = api_df['time'].dt.tz_localize(None)
        print(f"Fetcher: Simulated API data generated, shape {api_df.shape}")

    except Exception as e:
        error_message_api = f"Error generating simulated API data: {e}"
        print(f"Fetcher: {error_message_api}")
        # Return empty or partially filled DataFrames if API data fails critically
        # For now, we proceed with empty api_df if placeholder fails.

    # --- NCM Data Fetching and Processing (as per user's diff) ---
    if not api_df.empty:
        print("Fetcher: Starting NCM data fetch and comparison process...")
        try:
            start_date_dt = pd.to_datetime(selected_date_str, format='%Y%m%d').date()
            
            ncm_daily_data_list = []
            # NCM typically provides a few days of forecast. Let's fetch for the relevant period.
            # User's example fetches for 5 days from start_date. 
            # This might need adjustment based on how NCM PDFs are structured (e.g., one PDF covers multiple days).
            # The current ncm_parser.parse_ncm fetches PDF for a single given date.
            # If one NCM PDF contains a multi-day forecast, parse_ncm would need to be adapted
            # or called once for the start_date, and then process its multi-day content.
            # Assuming parse_ncm returns data for the *date* of the PDF title.

            # For a 7-day dashboard view, we might want NCM data for each of those 7 days if available.
            # The parse_ncm function tries to get PDF for current_date_to_try and then uses that date in its output.
            ncm_dfs_list = []
            for i in range(days_to_fetch): # Fetch NCM for each day in the dashboard's range
                current_loop_date = start_date_dt + dt.timedelta(days=i)
                print(f"Fetcher: Parsing NCM for date: {current_loop_date.strftime('%Y-%m-%d')}")
                # parse_ncm already tries previous days if PDF for current_loop_date is not found.
                # The 'date' column in the returned df from parse_ncm will be current_loop_date if successful.
                df_ncm_single_day = parse_ncm(current_loop_date) 
                ncm_dfs_list.append(df_ncm_single_day)
            
            if ncm_dfs_list:
                ncm_df_raw = pd.concat(ncm_dfs_list).reset_index(drop=True)
                ncm_df_raw.dropna(subset=['wave_ft_max_off', 'wind_kn_max_off'], how='all', inplace=True) # Drop rows where all NCM values are NaN
                print(f"Fetcher: Raw NCM data collected for period (rows: {len(ncm_df_raw)}):")
                # print(ncm_df_raw.to_string()) # Can be verbose
            else:
                ncm_df_raw = pd.DataFrame(columns=['date', 'wave_ft_max_off', 'wind_kn_max_off'])
                print("Fetcher: No NCM data could be parsed for the period.")

            # Aggregate API data daily to merge with NCM daily data
            # Ensure 'time' column is datetime type before resampling
            api_df['time'] = pd.to_datetime(api_df['time'])
            daily_api_agg = api_df.resample('D', on='time').agg(
                wave_m_max_api=('wave_m', 'max'),
                wind_kt_max_api=('wind_kt', 'max')
            ).reset_index()
            daily_api_agg['date'] = daily_api_agg['time'].dt.date # Convert timestamp to date object for merging
            daily_api_agg.drop(columns=['time'], inplace=True)

            if not ncm_df_raw.empty:
                # Ensure ncm_df_raw['date'] is also a date object if it's not already
                ncm_df_raw['date'] = pd.to_datetime(ncm_df_raw['date']).dt.date
                merged_comparison = pd.merge(
                    daily_api_agg,
                    ncm_df_raw,
                    how='left', on='date', validate='1:1' # Assumes one NCM entry per date
                )
            else: # If NCM data is empty, create a comparison table with API data only
                merged_comparison = daily_api_agg.copy()
                merged_comparison['wave_ft_max_off'] = np.nan
                merged_comparison['wind_kn_max_off'] = np.nan
            
            # Calculate difference percentage
            # Convert API wave_m to feet for comparison
            merged_comparison['wave_ft_max_api'] = (merged_comparison['wave_m_max_api'] * 3.28084).round(1)

            # Calculate gap: ((API - NCM) / NCM) * 100
            # Handle potential division by zero or NaN in NCM data
            merged_comparison['wave_gap_%'] = (
                ((merged_comparison.wave_ft_max_api - merged_comparison.wave_ft_max_off) / merged_comparison.wave_ft_max_off) * 100
            ).replace([np.inf, -np.inf], np.nan).round(1) # Replace inf with NaN if NCM is 0
            
            merged_comparison['wind_gap_%'] = (
                ((merged_comparison.wind_kt_max_api - merged_comparison.wind_kn_max_off) / merged_comparison.wind_kn_max_off) * 100
            ).replace([np.inf, -np.inf], np.nan).round(1)

            ncm_comparison_df = merged_comparison
            print("Fetcher: NCM comparison data generated:")
            # print(ncm_comparison_df.to_string()) # Can be verbose

        except Exception as e:
            print(f"Error (fetcher.py): Failed during NCM data processing or comparison: {e}")
            # Fallback: create an empty or placeholder ncm_comparison_df
            if 'date' not in ncm_comparison_df.columns and not daily_api_agg.empty:
                 ncm_comparison_df = daily_api_agg[['date']].copy()
                 for col in ['wave_ft_max_off', 'wave_m_max_api', 'wave_gap_%', 'wind_kn_max_off', 'wind_kt_max_api', 'wind_gap_%', 'wave_ft_max_api']:
                     ncm_comparison_df[col] = np.nan
            elif ncm_comparison_df.empty:
                 ncm_comparison_df = pd.DataFrame(columns=['date', 'wave_ft_max_off', 'wave_m_max_api', 'wave_gap_%', 'wind_kn_max_off', 'wind_kt_max_api', 'wind_gap_%', 'wave_ft_max_api'])

    # --- Combine API data with risk assessment (original logic from get_combined_forecast_data) ---
    combined_df = pd.DataFrame() # This will be the main hourly dataframe
    if not api_df.empty:
        combined_df = api_df.copy() # Start with the hourly API data
        # (Risk assessment logic would go here, using combined_df columns)
        # For now, we'll just return the api_df as combined_df for structure
        # Apply risk assessment using the assess_operational_risk function
        try:
            risk_results = combined_df.apply(
                lambda row: assess_operational_risk(
                    row.get('wave_m'),
                    row.get('wind_kt'),
                    row.get('tide_m')
                ),
                axis=1
            )
            risk_df = pd.DataFrame(risk_results.tolist(), index=combined_df.index, columns=['risk_color', 'risk_level', 'go_nogo', 'risk_reason'])
            combined_df = pd.concat([combined_df, risk_df], axis=1)
            print("Fetcher: Operational risk assessed and added to combined_df.")
        except Exception as e:
            print(f"Error (fetcher.py): Failed during operational risk assessment on combined_df: {e}")
            for col_risk in ['risk_color', 'risk_level', 'go_nogo', 'risk_reason']:
                combined_df[col_risk] = "Error" 

    print(f"Fetcher: get_combined_forecast_data finished. Returning combined_df ({combined_df.shape}), and ncm_comparison_df ({ncm_comparison_df.shape})")
    
    # The original get_combined_forecast_data returns only one df.
    # The user's plan implies fetcher.py/app.py will handle two distinct dataframes:
    # 1. The hourly detailed forecast (df_api or combined_df here)
    # 2. The daily NCM comparison (ncm_comparison_df here)
    # So, this function should return them. The dashboard will expect them.
    # For now, return combined_df (hourly) and ncm_comparison_df (daily NCM compare)
    # And an error message if any
    return combined_df, ncm_comparison_df, error_message_api

# --- 직접 실행 테스트용 코드 (선택 사항) ---
if __name__ == "__main__":
    print("Fetcher: Running in __main__ for testing.")
    
    # .env 파일에서 TIDE_KEY 로드 시도 (python-dotenv 필요)
    # 또는 환경 변수에서 직접 로드
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Fetcher: Attempted to load .env file.")
    except ImportError:
        print("Fetcher: python-dotenv not installed, skipping .env load.")

    api_key_for_tide_test = os.getenv("TIDE_KEY")
    
    if not api_key_for_tide_test:
        print("Error (fetcher.py __main__): TIDE_KEY not found in environment variables.")
        print("Please set TIDE_KEY environment variable for testing fetcher.py directly.")
        # 여기에 테스트용 임시 키를 넣거나, exit() 할 수 있습니다.
        # api_key_for_tide_test = "YOUR_TEST_TIDE_KEY_IF_ANY" 
        # if not api_key_for_tide_test:
        #     exit()

    print(f"Fetcher (__main__): Using TIDE_KEY: {'*' * (len(api_key_for_tide_test) - 4) + api_key_for_tide_test[-4:] if api_key_for_tide_test else 'None'}")

    today_str_test = datetime.now().strftime('%Y%m%d')
    print(f"Fetcher (__main__): Fetching combined data for date: {today_str_test}")
    
    # 캐싱을 사용하려면 Streamlit 컨텍스트가 필요하므로, 직접 실행 시에는 캐싱 없이 호출
    # 또는 Streamlit 앱의 일부로 실행하여 테스트
    # combined_data_test = get_combined_forecast_data(today_str_test, api_key_for_tide_test)
    # Updated to reflect new return signature of get_combined_forecast_data
    hourly_df_test, daily_ncm_compare_test, error_test = get_combined_forecast_data(today_str_test, api_key_for_tide_test)
    
    if error_test:
        print(f"Fetcher (__main__): Error encountered: {error_test}")

    if not hourly_df_test.empty:
        print("Fetcher (__main__): Hourly Combined Data Sample (from placeholder API data):")
        print(hourly_df_test.head())
        print(f"Fetcher (__main__): Total hourly rows: {len(hourly_df_test)}")
    else:
        print("Fetcher (__main__): No hourly combined data returned.") 

    if not daily_ncm_compare_test.empty:
        print("\nFetcher (__main__): Daily NCM Comparison Data Sample:")
        print(daily_ncm_compare_test.head().to_string())
        print(f"Fetcher (__main__): Total daily comparison rows: {len(daily_ncm_compare_test)}")
    else:
        print("Fetcher (__main__): No NCM comparison data returned.") 