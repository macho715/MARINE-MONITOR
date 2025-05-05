#!/usr/bin/env python3
import os, time, logging, datetime as dt, requests, pandas as pd, pytz
from requests.adapters import HTTPAdapter, Retry

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


def fetch_and_process_data():
    """Fetches 7-day data, processes it, generates tables, and returns results."""
    s = build_sess()
    df_combined = pd.DataFrame()
    trend_data = []
    daily_data = []
    error_message = None

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

        logging.info("Data fetched successfully.")

        marine_df = pd.DataFrame(marine_data["hourly"])
        wind_df   = pd.DataFrame(wind_data["hourly"])

        # Check for empty dataframes early
        if marine_df.empty or wind_df.empty:
            logging.warning("Received empty dataframe from one or both APIs.")
            error_message = "⚠️ Received empty data from API."
            return df_combined, trend_data, daily_data, error_message

        # Convert time and merge
        marine_df["time"] = pd.to_datetime(marine_df["time"])
        wind_df["time"]   = pd.to_datetime(wind_df["time"])
        df_combined = pd.merge(marine_df, wind_df, on="time", how="inner") # Use inner merge

        if df_combined.empty:
             logging.warning("Merge resulted in empty dataframe.")
             error_message = "⚠️ Merge resulted in empty data."
             return df_combined, trend_data, daily_data, error_message

        # Calculate wind knots and rename wave height column
        df_combined["wind_kt"] = df_combined[WEATHER_VARS] * 1.94384
        df_combined = df_combined.rename(columns={MARINE_VARS: "wave_m"})

        # Keep only necessary columns
        df_combined = df_combined[["time", "wave_m", "wind_kt"]].sort_values(by="time").reset_index(drop=True)

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