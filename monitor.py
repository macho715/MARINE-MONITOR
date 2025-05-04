#!/usr/bin/env python3
import os, time, logging, datetime as dt, requests, pandas as pd, pytz, schedule
from requests.adapters import HTTPAdapter, Retry

# Remove leading indentation from global variables
LAT, LON = 24.541664, 54.29167
MARINE_VARS = "wave_height,wave_direction,wind_wave_height" # Only valid marine variables
WEATHER_VARS = "wind_speed_10m" # Weather variable
THR_WAVE, THR_WIND = 2.0, 12.0 # Limits
UAE = pytz.timezone("Asia/Dubai")
DATA_DIR = "data"

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

def marine_req(s):
    """Fetches marine data from the Open-Meteo Marine API."""
    url = "https://marine-api.open-meteo.com/v1/marine"
    p = dict(latitude=LAT, longitude=LON, hourly=MARINE_VARS, # Uses correct MARINE_VARS
             forecast_days=2, timezone="auto", cell_selection="sea")
    r = s.get(url, params=p, timeout=10)
    r.raise_for_status()
    return r.json()

def wind_req(s):
    """Fetches wind data from the Open-Meteo Forecast API."""
    url = "https://api.open-meteo.com/v1/forecast"
    p = dict(latitude=LAT, longitude=LON, hourly=WEATHER_VARS, # Uses correct WEATHER_VARS
             forecast_days=2, timezone="auto")
    r = s.get(url, params=p, timeout=10)
    r.raise_for_status()
    return r.json()

def job():
    """The main job to fetch, process, and save the data."""
    s = build_sess()
    try:
        logging.info("Fetching marine and wind data...")
        marine_data = marine_req(s) # Correct function call
        wind_data = wind_req(s)     # Correct function call
        logging.info("Data fetched successfully.")

        marine_df = pd.DataFrame(marine_data["hourly"])
        wind_df   = pd.DataFrame(wind_data["hourly"])

        # Ensure time columns are datetime objects before merging
        marine_df["time"] = pd.to_datetime(marine_df["time"])
        wind_df["time"]   = pd.to_datetime(wind_df["time"])

        # Merge the dataframes using an outer join to keep all time points
        df = pd.merge(marine_df, wind_df[["time", "wind_speed_10m"]], on="time", how="outer", suffixes=('_marine', '_wind'))

        if df.empty:
             logging.warning("Received empty dataframes or merge resulted in empty dataframe.")
             return

        # Sort by time just in case the merge disordered it
        df = df.sort_values(by="time").reset_index(drop=True)

        # Get the latest complete row where both wave and wind are available
        latest_complete = df.dropna(subset=["wave_height", "wind_speed_10m"]).iloc[-1] if not df.dropna(subset=["wave_height", "wind_speed_10m"]).empty else None

        if latest_complete is not None:
            status = ("✅ OPERATE" if latest_complete.wave_height <= THR_WAVE
                      and latest_complete.wind_speed_10m <= THR_WIND else "⛔ HOLD")
            logging.info("%s | SWH %.2f m | Wind %.1f m/s → %s",
                         latest_complete.time.strftime('%Y-%m-%d %H:%M'), latest_complete.wave_height, latest_complete.wind_speed_10m, status)
        else:
            # Log the latest available data even if incomplete
            latest_any = df.iloc[-1]
            status = "⚠️ INCOMPLETE DATA"
            logging.warning("%s | Data potentially incomplete. SWH: %.2f m, Wind: %.1f m/s → %s",
                          latest_any.time.strftime('%Y-%m-%d %H:%M'),
                          latest_any.wave_height if pd.notna(latest_any.wave_height) else -1,
                          latest_any.wind_speed_10m if pd.notna(latest_any.wind_speed_10m) else -1,
                          status)

        # Save the merged dataframe
        fn = dt.datetime.now(UAE).strftime("%Y%m%d_%H%M") + ".csv"
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, fn)
        df.to_csv(file_path, index=False)
        logging.info(f"Combined forecast data saved to {file_path}")

    except requests.exceptions.RequestException as exc:
        logging.exception("API fetch failed: %s", exc)
        if exc.response is not None:
            logging.error(f"API Response Status: {exc.response.status_code}")
            logging.error(f"API Response Body: {exc.response.text}")
    except KeyError as e:
        logging.exception(f"Missing expected key in API response: {e}.")
    except Exception as e:
        logging.exception(f"An error occurred during job execution: {e}")

if __name__ == "__main__":
    # Ensure correct indentation for this block
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.info("Marine monitor started (runs at 08:00 / 20:00 GST)")
    # Define schedule using the correct timezone object
    schedule.every().day.at("08:00", UAE).do(job)
    schedule.every().day.at("20:00", UAE).do(job)
    # Run job once immediately on startup
    job()
    while True:
        # Ensure correct indentation for the loop contents
        schedule.run_pending()
        time.sleep(30) # Check every 30 seconds