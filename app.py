import streamlit as st
import pandas as pd
import logging
import altair as alt # Import Altair
# Import only necessary items from fetcher
from fetcher import fetch_and_process_data, LAT, LON#, THR_WAVE, THR_WIND # Thresholds no longer needed here

# --- Simple Authentication ---
def check_password():
    """Returns True if the user entered the correct password."""
    
    # Check if APP_KEY is set in secrets
    if "APP_KEY" not in st.secrets or not st.secrets["APP_KEY"]:
        # If no APP_KEY is set, bypass authentication
        return True
        
    # If APP_KEY is set, require password
    password = st.text_input("🔐 Access Key", type="password")
    if not password: # If password input is empty, stop
        st.warning("Please enter the access key.")
        st.stop()
        
    if password == st.secrets["APP_KEY"]:
        return True
    else:
        st.error("🚨 Access key is incorrect.")
        st.stop()
        return False

# Check password at the very beginning
if not check_password():
    st.stop() # Stop execution if password check fails
# --- End Simple Authentication ---

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Set page config for wider layout
st.set_page_config(layout="wide")

st.title("🌊 Marine Weather Monitor (7-Day Forecast)") # Updated title
st.caption(f"Monitoring conditions near Lat: {LAT}, Lon: {LON}")

# Thresholds are now implicitly defined in the tables below
# col1, col2 = st.columns(2)
# with col1:
#     st.metric("Wave Height Threshold (SWH)", f"{THR_WAVE:.1f} m")
# with col2:
#     st.metric("Wind Speed Threshold (10m)", f"{THR_WIND:.1f} m/s")

st.markdown("---") # Separator

# Use caching to avoid refetching data on every interaction
# TTL = Time To Live - cache expires after 30 minutes (1800 seconds)
@st.cache_data(ttl=1800)
def get_data(tide_api_key): # Add tide_api_key parameter
    logging.info("Cache miss or expired. Fetching new data.")
    # Pass the key to the fetcher function
    df, trend_data, daily_data, error_message = fetch_and_process_data(tide_api_key=tide_api_key)
    return df, trend_data, daily_data, error_message

# Button to manually refresh data (clears cache)
if st.button("🔄 Refresh Data"):
    st.cache_data.clear() # Clear the cache to force a refresh

# --- Get Tide API Key from Secrets ---
tide_key = st.secrets.get("TIDE_KEY", None)
if not tide_key:
    st.sidebar.warning("WorldTides API Key (TIDE_KEY) not found in secrets. Tide data will not be fetched.", icon="🔑")
# --- End Get Tide API Key ---

# Fetch data (uses cache if available and not expired)
st.subheader("Forecast Summary & Sailing Decision")
with st.spinner('Fetching latest 7-day forecast data...'):
    try:
        # Pass the tide key to get_data
        df, trend_data, daily_data, error_message = get_data(tide_api_key=tide_key)

        # Display error message if fetching/processing failed
        if error_message:
            st.error(error_message)
        else:
            # --- Add Altair Chart --- 
            if not df.empty:
                st.subheader("7-Day Forecast Trend Chart")
                
                # --- Simple Altair Chart for Debugging ---
                try:
                    simple_chart = alt.Chart(df).mark_line().encode(
                        x='time:T',
                        y='wave_m:Q',
                        tooltip=['time:T', 'wave_m:Q']
                    ).properties(
                        title='DEBUG: Wave Height Only'
                    )
                    st.altair_chart(simple_chart, use_container_width=True)
                    st.success("DEBUG: Simple chart rendering attempted.")
                except Exception as chart_err:
                    st.error(f"DEBUG: Error rendering simple chart: {chart_err}")
                # --- End Simple Altair Chart ---

                # --- Original Complex Chart Code (Commented Out for Debugging) ---
                # base = alt.Chart(df).encode(x='time:T')
                # line_actual = base.transform_fold(...).mark_line(...).encode(...)
                # line_forecast = alt.LayerChart()
                # if 'wave_pred' in df.columns and df['wave_pred'].notna().any():
                #     line_forecast = base.mark_line(...).encode(...)
                # line_tide = alt.LayerChart()
                # if 'tide_m' in df.columns and df['tide_m'].notna().any():
                #     line_tide = base.mark_line(...).encode(...)
                #     layered_chart = alt.layer(line_actual, line_forecast, line_tide).resolve_scale(y='independent')
                #     st.altair_chart(layered_chart, use_container_width=True)
                # else:
                #     layered_chart = alt.layer(line_actual, line_forecast)
                #     st.altair_chart(layered_chart, use_container_width=True)
                # --- End Original Complex Chart Code ---
                
                st.markdown("---") # Separator after chart
            # --- End Altair Chart ---

            # Display Trend Table
            st.markdown("**2-Day & 7-Day 트렌드 요약표 (단위: m / kt)**")
            if trend_data:
                trend_df = pd.DataFrame(trend_data, columns=["Day (LT)", "모드", "파고", "풍속", "위험등급", "작업 권고"])
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Trend data could not be generated.")

            st.markdown("---") # Separator

            # Display Daily Sailing Table
            st.markdown("**일별 항해 가능 여부 (Al Ghallan→Abu Dhabi)**")
            if daily_data:
                daily_df = pd.DataFrame(daily_data, columns=["Date", "00-12 LT Wind / Wave*", "12-24 LT Wind / Wave*", "Risk", "Decision", "참고"])
                st.dataframe(daily_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Daily sailing data could not be generated.")

            st.markdown("---") # Separator

            # Display Raw Hourly Data (Optional, can be commented out if not needed)
            st.subheader("7-Day Hourly Forecast Data (Raw)")
            if not df.empty:
                # Make time column more readable for display
                df_display = df.copy()
                df_display['time'] = pd.to_datetime(df_display['time']).dt.strftime('%Y-%m-%d %H:%M')
                df_display['wave_m'] = df_display['wave_m'].round(1) # Round wave height
                df_display['wind_kt'] = df_display['wind_kt'].round(0).astype(int) # Round wind speed
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # Add CSV download button for the raw data
                csv_data = df.to_csv(index=False).encode('utf-8') # Prepare data for download
                st.download_button(
                    label="📥 Download Raw Data as CSV",
                    data=csv_data,
                    file_name='7day_forecast_raw.csv',
                    mime='text/csv',
                )
            else:
                st.warning("Raw hourly data is not available.")

    except Exception as e:
        st.error(f"An error occurred in the Streamlit app: {e}")
        logging.exception("Error in Streamlit app display section")

# Add a small footer
st.markdown("---")
st.caption("Data from Open-Meteo Marine & Forecast APIs.") 