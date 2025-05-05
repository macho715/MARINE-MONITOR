import streamlit as st
import pandas as pd
import logging
# Import only necessary items from fetcher
from fetcher import fetch_and_process_data, LAT, LON#, THR_WAVE, THR_WIND # Thresholds no longer needed here

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
# TTL = Time To Live - cache expires after 10 minutes (600 seconds)
@st.cache_data(ttl=600)
def get_data():
    logging.info("Cache miss or expired. Fetching new data.")
    # Unpack the new return values from the updated fetcher function
    df, trend_data, daily_data, error_message = fetch_and_process_data()
    return df, trend_data, daily_data, error_message

# Button to manually refresh data (clears cache)
if st.button("🔄 Refresh Data"):
    st.cache_data.clear() # Clear the cache to force a refresh

# Fetch data (uses cache if available and not expired)
st.subheader("Forecast Summary & Sailing Decision")
with st.spinner('Fetching latest 7-day forecast data...'):
    try:
        df, trend_data, daily_data, error_message = get_data()

        # Display error message if fetching/processing failed
        if error_message:
            st.error(error_message)
        else:
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