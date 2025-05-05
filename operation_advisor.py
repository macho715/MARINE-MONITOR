#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marine + Weather 예보(3~16 일)로 LOLO·RORO·항해 가능 여부를 판정하고
CSV 저장·콘솔 출력하며, 그래프 생성 및 Slack/이메일 알림을 전송하고,
매일 지정된 시간에 자동 실행되는 통합 스크립트 (임계치 YAML 로드).
"""

import os, time, logging, datetime as dt, smtplib, yaml # Added yaml
import requests, pandas as pd
import matplotlib.pyplot as plt, schedule, pytz
from email.message import EmailMessage
from requests.adapters import HTTPAdapter, Retry
from slack_sdk.webhook import WebhookClient
from dotenv import load_dotenv

# ── 환경 설정 ─────────────────────────────────────
load_dotenv()
LAT = float(os.getenv("LAT", 24.55))
LON = float(os.getenv("LON", 54.30))
SLACK_URL = os.getenv("SLACK_WEBHOOK_URL")
MAIL_TO   = os.getenv("MAIL_TO")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

FORECAST_DAYS = 3 # 16까지 확대 가능
tz = pytz.timezone("Asia/Dubai")
DATA_DIR = "data"; os.makedirs(DATA_DIR, exist_ok=True)
PLOT_FILENAME = "plot.png"

# ── 임계치 로드 ────────────────────────────────────
try:
    with open("thresholds.yaml", "r", encoding="utf-8") as f:
        THR = yaml.safe_load(f)
    logging.info(f"Loaded thresholds from thresholds.yaml: {THR}")
except FileNotFoundError:
    logging.error("thresholds.yaml not found. Using default thresholds.")
    # Define default thresholds here if the file is missing
    THR = {
        "LOLO": dict(swh=1.0, wind=10.0, swell=0.5, dir_tol=60),
        "RORO": dict(swh=0.6, wind=8.0,  swell=0.3, dir_tol=30),
        "SAIL": dict(swh=2.0, wind=12.0)
    }
except yaml.YAMLError as e:
     logging.error(f"Error parsing thresholds.yaml: {e}. Using default thresholds.")
     THR = {
        "LOLO": dict(swh=1.0, wind=10.0, swell=0.5, dir_tol=60),
        "RORO": dict(swh=0.6, wind=8.0,  swell=0.3, dir_tol=30),
        "SAIL": dict(swh=2.0, wind=12.0)
    }

# ── 로깅 설정 ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── 공통 유틸 ─────────────────────────────────────
def build_session(retries=3):
    retry = Retry(total=retries, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    s = requests.Session();  s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

sess = build_session()

def fetch_json(url, params):
    try:
        r = sess.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API fetch failed for {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"API Status: {e.response.status_code}, Body: {e.response.text}")
        return None

def fetch_df(days=FORECAST_DAYS):
    marine_vars = "wave_height,wind_wave_height,swell_wave_height,wave_direction"
    weather_vars = "wind_speed_10m,wind_direction_10m"
    
    logging.info("Fetching marine and wind data...")
    marine_data = fetch_json(
        "https://marine-api.open-meteo.com/v1/marine",
        dict(latitude=LAT, longitude=LON, hourly=marine_vars,
             forecast_days=days, timezone="auto", cell_selection="sea")
    )
    weather_data = fetch_json(
        "https://api.open-meteo.com/v1/forecast",
        dict(latitude=LAT, longitude=LON, hourly=weather_vars,
             forecast_days=days, timezone="auto")
    )

    if not marine_data or "hourly" not in marine_data or not weather_data or "hourly" not in weather_data:
        logging.error("Failed to retrieve valid data from one or both APIs.")
        return None

    try:
        m_df = pd.DataFrame(marine_data["hourly"])
        w_df = pd.DataFrame(weather_data["hourly"])
        for d in (m_df, w_df): d["time"] = pd.to_datetime(d["time"])
        df = pd.merge(m_df, w_df[["time", "wind_speed_10m", "wind_direction_10m"]], on="time", how="outer")
        df = df.sort_values(by="time").reset_index(drop=True)
        logging.info("Data fetched and merged successfully.")
        return df
    except Exception as e:
        logging.error(f"Error processing or merging data: {e}")
        return None

# ── 판정 로직 ────────────────────────────────────
def judge(row, lim):
    # Ensure required columns exist and handle potential NaNs
    swh_ok = pd.notna(row.get('wave_height')) and row['wave_height'] <= lim['swh']
    wind_ok = pd.notna(row.get('wind_speed_10m')) and row['wind_speed_10m'] <= lim['wind']
    swell_ok = pd.notna(row.get('swell_wave_height')) and row['swell_wave_height'] <= lim.get('swell', 99)
    
    ok = swh_ok and wind_ok and swell_ok
    
    # Check direction tolerance if applicable and data is available
    if ok and 'dir_tol' in lim and pd.notna(row.get('wave_direction')) and pd.notna(row.get('wind_direction_10m')):
        rel = abs((row['wave_direction'] - row['wind_direction_10m'] + 360)) % 360
        ok &= (rel <= lim['dir_tol'] or rel >= 360 - lim['dir_tol'])
        
    return "OK" if ok else "HOLD"

def evaluate(df):
    if df is None or df.empty:
        return df
    for k, lim in THR.items():
        try:
            # Ensure required columns exist before applying judgement
            required_cols = ['wave_height', 'wind_speed_10m', 'swell_wave_height']
            if 'dir_tol' in lim:
                 required_cols.extend(['wave_direction', 'wind_direction_10m'])
            
            if not all(col in df.columns for col in required_cols):
                 logging.warning(f"Missing required columns for {k} judgment. Skipping.")
                 df[f"{k}_status"] = "N/A"
                 continue
                 
            df[f"{k}_status"] = df.apply(judge, axis=1, lim=lim)
        except Exception as e:
            logging.error(f"Error applying judgment for {k}: {e}")
            df[f"{k}_status"] = "ERROR"
    return df

# ── 그래프 & 알림 함수 ──────────────────────────────
def save_plot(df, fp):
    if df is None or df.empty:
        logging.warning("Cannot generate plot, DataFrame is empty.")
        return False
    try:
        plt.figure(figsize=(10, 4))
        # Check if columns exist before plotting
        if 'wave_height' in df.columns: plt.plot(df["time"], df["wave_height"], label="SWH (m)", marker='.')
        if 'wind_speed_10m' in df.columns: plt.plot(df["time"], df["wind_speed_10m"], label="Wind 10 m (m/s)", marker='.')
        # Add threshold lines using loaded THR dictionary
        if 'SAIL' in THR:
            if 'swh' in THR['SAIL']: plt.axhline(THR["SAIL"]["swh"], color='orange', ls="--", lw=1, label=f"Sail SWH Limit ({THR['SAIL']['swh']}m)")
            if 'wind' in THR['SAIL']: plt.axhline(THR["SAIL"]["wind"], color='red', ls=":", lw=1, label=f"Sail Wind Limit ({THR['SAIL']['wind']}m/s)")
        plt.legend()
        plt.title(f"Al Ghallan Marine Forecast ({LAT},{LON})")
        plt.xlabel("Time (Local)"); plt.ylabel("Value")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(fp); plt.close()
        logging.info(f"Plot saved to {fp}")
        return True
    except Exception as e:
        logging.error(f"Failed to generate or save plot: {e}")
        return False

def send_slack(msg, img):
    if not SLACK_URL:
        logging.info("SLACK_WEBHOOK_URL not set. Skipping Slack notification.")
        return
    try:
        client = WebhookClient(SLACK_URL)
        if os.path.exists(img):
            with open(img,"rb") as f: client.send(text=msg, filename="forecast.png", file=f)
            logging.info("Slack notification with plot sent.")
        else:
             client.send(text=msg)
             logging.info("Slack notification sent (plot missing).")
    except Exception as e:
        logging.exception(f"Error sending Slack notification: {e}")

def send_mail(subject, body, img):
    if not MAIL_TO or not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        logging.info("Email settings not fully configured. Skipping email notification.")
        return
    msg = EmailMessage(); msg["Subject"]=subject; msg["From"]=SMTP_USER; msg["To"]=MAIL_TO
    msg.set_content(body.replace("*","")) # Plain text fallback
    html_body = body.replace("\n", "<br>").replace("*", "<b>") + "</b>"
    msg.add_alternative(f"<html><body>{html_body}</body></html>", subtype='html')
    if os.path.exists(img):
        try:
            with open(img,"rb") as f: msg.add_attachment(f.read(), maintype="image", subtype="png", filename=os.path.basename(img))
        except Exception as e:
            logging.error(f"Failed to attach plot to email: {e}")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
            logging.info(f"Email notification sent successfully to {MAIL_TO}.")
    except smtplib.SMTPAuthenticationError:
         logging.error("SMTP Authentication failed. Check credentials.")
    except Exception as e:
        logging.exception(f"Error sending email notification: {e}")

# ── 스케줄 작업 ──────────────────────────────────
def job():
    logging.info("Fetch & evaluate job started...")
    df = fetch_df()
    if df is None:
        logging.error("Job failed: Could not fetch or process data.")
        return
        
    df = evaluate(df)
    
    if df.empty:
        logging.warning("Job warning: DataFrame is empty after evaluation.")
        return

    # Ensure required columns exist before accessing them
    last_row_valid = ('LOLO_status' in df.columns and 'RORO_status' in df.columns and 'SAIL_status' in df.columns)
    if last_row_valid:
        last = df.iloc[-1]
        status_summary = f"LOLO:{last.LOLO_status} RORO:{last.RORO_status} SAIL:{last.SAIL_status}"
        msg = (
            f"*Al Ghallan Marine Update* ({last.time:%Y-%m-%d %H:%M} Local)\n"
            f"> SWH: {last.wave_height:.2f} m | Wind: {last.wind_speed_10m:.1f} m/s | Wave Dir: {last.wave_direction:.0f}°\n"
            f"> Status: {status_summary}"
        )
        email_subject = f"Al Ghallan Forecast Update - {last.time:%Y-%m-%d %H:%M}"
        
        # Log the summary
        logging.info(
            "Latest (%s) | SWH %.2f m | Wind %.1f m/s | Dir Wave %.0f° Wind %.0f° → %s",
            last.time.strftime("%Y-%m-%d %H:%M"),
            last.wave_height if pd.notna(last.wave_height) else -1,
            last.wind_speed_10m if pd.notna(last.wind_speed_10m) else -1,
            last.wave_direction if pd.notna(last.wave_direction) else -1,
            last.wind_direction_10m if pd.notna(last.wind_direction_10m) else -1,
            status_summary
        )

        plot_saved = save_plot(df, PLOT_FILENAME)
        
        if plot_saved:
            send_slack(msg, PLOT_FILENAME)
            send_mail(email_subject, msg, PLOT_FILENAME)
        else:
            logging.warning("Proceeding with notifications without plot attachment.")
            send_slack(msg, "")
            send_mail(email_subject, msg, "")
    else:
         logging.warning("Could not generate status summary or send notifications due to missing status columns in the DataFrame.")

    # CSV 저장
    try:
        fn = dt.datetime.now(tz).strftime("%Y%m%d_%H%M") + "_operations.yaml_based.csv"
        csv_path = os.path.join(DATA_DIR, fn)
        df.to_csv(csv_path, index=False)
        logging.info(f"Combined forecast and status data saved to {csv_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
    logging.info("Job finished.")

# ── 스케줄러 실행 ────────────────────────────────
schedule.every().day.at("08:00", tz).do(job)
schedule.every().day.at("20:00", tz).do(job)

if __name__ == "__main__":
    # Add PyYAML to requirements if not already there
    # pip install PyYAML
    logging.info("Scheduler started (runs at 08:00/20:00 GST)")
    job() # Run once immediately
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            logging.info("Scheduler stopped manually.")
            break
        except Exception as e:
            logging.exception("Error in scheduler loop: %s", e)
            time.sleep(60) 