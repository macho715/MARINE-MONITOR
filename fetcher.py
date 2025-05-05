#!/usr/bin/env python3
import os, time, logging, datetime as dt, requests, pandas as pd, pytz
from requests.adapters import HTTPAdapter, Retry
from statsmodels.tsa.arima.model import ARIMA
from pandas.errors import OutOfBoundsDatetime
import streamlit as st # Keep streamlit import if needed for secrets in app.py

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

def fetch_and_process_data(tide_api_key=None):
    """Fetches 7-day data, processes it, generates tables & forecast, and returns results."""
    s = build_sess()
    df_combined = pd.DataFrame()
    trend_data = []
    daily_data = []
    error_message = None
    df_forecast = pd.DataFrame() # Initialize forecast dataframe

    try:
        logging.info(f"Fetching {DAYS}-day marine and wind data...")
        # Fetch Marine Data
        m_url = "https://marine-api.open-meteo.com/v1/marine"
        m_params = dict(latitude=LAT, longitude=LON, hourly=MARINE_VARS,
                        forecast_days=DAYS, timezone=TZ_AUTO, cell_selection="sea")
        m_resp = s.get(m_url, params=m_params, timeout=10)
        m_resp.raise_for_status()
        marine_data = m_resp.json()

        # Fetch Weather Data
        w_url = "https://api.open-meteo.com/v1/forecast"
        w_params = dict(latitude=LAT, longitude=LON, hourly=WEATHER_VARS,
                        forecast_days=DAYS, timezone=TZ_AUTO)
        w_resp = s.get(w_url, params=w_params, timeout=10)
        w_resp.raise_for_status()
        wind_data = w_resp.json()

        # Fetch Tide Data
        logging.info(f"Fetching {DAYS}-day tide data...")
        tide_df = fetch_tide(LAT, LON, tide_api_key, days=DAYS)
        if not tide_df.empty:
            logging.info("Tide data fetched successfully.")
        
        logging.info("Data fetched successfully.")

        marine_df = pd.DataFrame(marine_data["hourly"])
        wind_df   = pd.DataFrame(wind_data["hourly"])

        # Check for empty dataframes early
        if marine_df.empty or wind_df.empty:
            logging.warning("Received empty dataframe from one or both APIs.")
            error_message = "⚠️ Received empty data from API."
            return df_combined, trend_data, daily_data, error_message

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
            return df_final, [], [], error_message

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
             return df_combined, trend_data, daily_data, error_message

        # Calculate wind knots and rename wave height column
        df_combined["wind_kt"] = df_combined[WEATHER_VARS] * 1.94384
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
                  
             if 'wave_m' in df_combined_indexed.columns:
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
    return df_combined, trend_data, daily_data, error_message

# Keep build_sess, remove other old functions if they are not used anymore
# Remove marine_req, wind_req as logic is now inside fetch_and_process_data

# Remove __main__ block as this is now a module
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, ...)
#     ... schedule logic ... 