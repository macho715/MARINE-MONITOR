#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetches and analyzes the D+1 and D+2 marine forecast 
(wave height, direction) for Al Ghallan Island (approx. 24.55 N, 54.30 E)
using the Open-Meteo Marine API.
"""

import requests
import pandas as pd
import datetime as dt

# --- Configuration ---
LAT, LON = 24.55, 54.30        # Al Ghallan Island coordinates
MARINE_VARS = "wave_height,wind_wave_height,wave_direction" # Variables to fetch
FORECAST_DAYS = 3              # Fetch today + next 2 days
API_TIMEOUT = 10               # Seconds
OUTPUT_CSV_FILE = "al_ghallan_Dplus1_Dplus2.csv"

# --- Functions ---

def fetch_marine_data(lat, lon, variables, days, timeout):
    """Fetches marine forecast data from Open-Meteo API."""
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat, 
        "longitude": lon,
        "hourly": variables,
        "forecast_days": days,
        "timezone": "auto", # Automatically detect local timezone
        "cell_selection": "sea" # Prefer sea points near coast
    }
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4XX, 5XX)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API Response Status: {e.response.status_code}")
            print(f"API Response Body: {e.response.text}")
        return None

def process_data(api_data):
    """Converts API JSON response to a pandas DataFrame and processes time."""
    if not api_data or "hourly" not in api_data:
        print("Error: Invalid or missing API data.")
        return None
    try:
        df = pd.DataFrame(api_data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"Error processing data into DataFrame: {e}")
        return None

def filter_and_analyze(df):
    """Filters for D+1, D+2, analyzes wave height, and saves to CSV."""
    if df is None:
        return
        
    base_date = dt.date.today()
    # Create a set of target dates (D+1 and D+2)
    target_dates = {base_date + dt.timedelta(days=i) for i in (1, 2)}
    
    # Filter the DataFrame for the target dates
    df_next2 = df[df["time"].dt.date.isin(target_dates)].copy()
    
    if df_next2.empty:
        print("No data found for the next two days (D+1, D+2).")
        return

    print("► Wave height summary for D+1 & D+2 (m):")
    # Group by date and get descriptive statistics for wave_height
    summary = df_next2.groupby(df_next2["time"].dt.date)["wave_height"].describe()[["min", "mean", "max"]]
    print(summary)

    # Save the filtered data to CSV
    try:
        df_next2.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nFiltered data for D+1 and D+2 saved to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Fetching marine forecast for Al Ghallan Island ({LAT} N, {LON} E)...")
    api_response = fetch_marine_data(LAT, LON, MARINE_VARS, FORECAST_DAYS, API_TIMEOUT)
    
    if api_response:
        dataframe = process_data(api_response)
        filter_and_analyze(dataframe)
    else:
        print("Failed to retrieve or process data. Exiting.") 