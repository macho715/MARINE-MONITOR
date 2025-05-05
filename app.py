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
            # --- DEBUGGING CODE START ---
            # st.subheader("(DEBUG) Data Input for Chart") # Keep commented for now
            # if not df.empty:
            #     st.dataframe(df.head())
            #     import io
            #     buffer = io.StringIO()
            #     df.info(buf=buffer)
            #     s = buffer.getvalue()
            #     st.text(s)
            # else:
            #     st.warning("DataFrame passed to chart is empty!")
            # st.markdown("---")
            # --- DEBUGGING CODE END ---

            # --- Add Checkbox Toggle Chart (Wave, Wind, Tide) ---
            st.subheader("7-Day Forecast Trend Chart") # Restore original subheader
            if not df.empty:
                try:
                    # --- 1) Checkboxes for Series Selection ---
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        show_wave = st.checkbox("🌊 SWH (m)", value=True)
                    with col2:
                        show_wind = st.checkbox("💨 Wind (kt)", value=True)
                    with col3:
                        # Only show tide checkbox if tide data is available
                        show_tide = False 
                        if 'tide_m' in df.columns and df['tide_m'].notna().any():
                            show_tide = st.checkbox("🌑 Tide (m)", value=False)
                    st.markdown("<hr style='margin-top:0; margin-bottom:1rem;'>", unsafe_allow_html=True)
                    
                    # --- 2) Define Altair Layers Conditionally ---
                    base = alt.Chart(df).encode(x=alt.X("time:T", axis=alt.Axis(title="Time"))) # Use simplified axis title from previous edit
                    
                    layers = []
                    
                    if show_wave:
                        wave = base.mark_line(color="#1f77b4", strokeWidth=1.5).encode(
                            y=alt.Y("wave_m:Q", axis=alt.Axis(title="SWH (m)", titleColor="#1f77b4")),
                            tooltip=['time:T', alt.Tooltip('wave_m:Q', format='.1f', title='SWH (m)')]
                        )
                        layers.append(wave)
                    
                    if show_wind:
                        wind = base.mark_line(color="#ff7f0e", strokeWidth=1.5, strokeDash=[4,3]).encode(
                            y=alt.Y("wind_kt:Q", axis=alt.Axis(title="Wind (kt)", titleColor="#ff7f0e")),
                            tooltip=['time:T', alt.Tooltip('wind_kt:Q', format='.0f', title='Wind (kt)')]
                        )
                        layers.append(wind)
                    
                    if show_tide and 'tide_m' in df.columns: # Double check column exists
                        tide = base.mark_line(color="#268bd2", strokeDash=[2,2]).encode(
                            y=alt.Y("tide_m:Q", axis=alt.Axis(title="Tide (m)", titleColor="#268bd2")),
                            tooltip=['time:T', alt.Tooltip('tide_m:Q', format='.1f', title='Tide (m)')]
                        )
                        layers.append(tide)
                        
                    # --- 3) Render Layered Chart ---
                    if layers:
                        chart = alt.layer(*layers).resolve_scale(y="independent").properties(height=350)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        # Display a message if no series are selected
                        st.info("위에 있는 체크박스를 선택하여 그래프에 표시할 데이터를 고르세요.") 
                        # Optionally display an empty chart area or specific placeholder
                        # placeholder_chart = alt.Chart(pd.DataFrame({'x':[df['time'].min(), df['time'].max()]})).mark_text().encode(x='x:T').properties(height=350)
                        # st.altair_chart(placeholder_chart, use_container_width=True)
                        
                except Exception as chart_err:
                    st.error(f"Error rendering checkbox chart: {chart_err}")
            else: # Handle case where df is empty initially
                  st.warning("No forecast data available to plot.")
            # --- End Checkbox Toggle Chart Section ---
                
            st.markdown("---") # Separator after chart

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