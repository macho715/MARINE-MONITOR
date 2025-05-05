import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import datetime as dt

DATA_DIR = Path("data") # Directory where CSV files are stored

# --- Page Configuration ---
st.set_page_config(page_title="Marine Forecast Dashboard", layout="wide")

# --- Helper Functions ---
def get_available_dates(data_dir):
    """Finds all unique dates from CSV filenames in the data directory."""
    dates = set()
    if data_dir.exists():
        # Extract YYYYMMDD part from filenames like YYYYMMDD_HHMM_operations.yaml_based.csv
        for f in data_dir.glob("*_operations*.csv"): 
            try:
                # Extract the date part reliably
                date_str = f.stem.split('_')[0]
                # Validate if it's a date
                dt.datetime.strptime(date_str, '%Y%m%d')
                dates.add(date_str)
            except (IndexError, ValueError):
                # Ignore files that don't match the expected format
                continue 
    return sorted(list(dates), reverse=True)

def load_latest_data_for_date(selected_date, data_dir):
    """Loads the most recent CSV file for a selected date."""
    files_for_date = sorted(data_dir.glob(f"{selected_date}*_operations*.csv"), reverse=True)
    if files_for_date:
        try:
            df = pd.read_csv(files_for_date[0])
            df["time"] = pd.to_datetime(df["time"])
            return df, files_for_date[0].name # Return filename as well
        except Exception as e:
            st.error(f"Error loading data from {files_for_date[0].name}: {e}")
            return pd.DataFrame(), None
    else:
        st.warning(f"No data files found for date {selected_date}.")
        return pd.DataFrame(), None

# --- Sidebar --- 
st.sidebar.title("Marine Dashboard")
available_dates = get_available_dates(DATA_DIR)

if not available_dates:
    st.sidebar.warning("No data files found in the 'data' directory.")
    st.warning("No data available. Please run the `operation_advisor.py` script first to generate data.")
    st.stop() # Stop execution if no data is available

selected_date_str = st.sidebar.selectbox("Select Date", available_dates)

# --- Main Content ---
df, loaded_filename = load_latest_data_for_date(selected_date_str, DATA_DIR)

if df.empty:
    st.stop() # Stop if data loading failed or returned empty df

# Display Title with Date and Filename
selected_dt = dt.datetime.strptime(selected_date_str, '%Y%m%d')
st.title(f"Al Ghallan Forecast Dashboard – {selected_dt.strftime('%Y-%m-%d')}")
st.caption(f"Data loaded from: `{loaded_filename}`")

# --- Data Display ---
st.subheader("Forecast Data Table")
st.dataframe(df)

# --- Charts ---
st.subheader("Forecast Charts")

# Chart 1: Wave Height and Wind Speed Lines
line_chart_data = df.melt(id_vars=["time"], value_vars=["wave_height", "wind_speed_10m"], var_name="Metric", value_name="Value")

line = alt.Chart(line_chart_data).mark_line(point=True).encode(
    x=alt.X("time:T", title="Time"),
    y=alt.Y("Value:Q", title="Value"),
    color=alt.Color("Metric:N", title="Metric", scale=alt.Scale(domain=["wave_height", "wind_speed_10m"], range=["#1f77b4", "#ff7f0e"])), # Blue for wave, Orange for wind
    tooltip=["time:T", "Metric:N", "Value:Q"]
).properties(
    height=300
).interactive() # Enable zooming and panning

st.altair_chart(line, use_container_width=True)

# Chart 2: Operational Status Heatmap
status_cols = [col for col in df.columns if col.endswith("_status")] # Find status columns

if status_cols:
    heatmap_data = df.melt(id_vars="time", value_vars=status_cols, var_name="Operation", value_name="Status")
    # Clean up operation names (remove _status)
    heatmap_data['Operation'] = heatmap_data['Operation'].str.replace('_status', '')
    
    heat = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X("time:T", title="Time", axis=alt.Axis(format="%H:%M")), # Show hours/minutes
        y=alt.Y("Operation:N", title="Operation Type"),
        color=alt.condition(
            alt.datum.Status == "OK", 
            alt.value("steelblue"), # Color for OK
            alt.value("orangered")   # Color for HOLD
        ),
        tooltip=["time:T", "Operation:N", "Status:N"]
    ).properties(
        title="Operational Status Timeline",
        height=120
    ).interactive()
    
    st.altair_chart(heat, use_container_width=True)
else:
    st.warning("No status columns found in the data for the heatmap.") 