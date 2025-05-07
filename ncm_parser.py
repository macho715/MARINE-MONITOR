# ncm_parser.py
import camelot, datetime as dt, pandas as pd, re, io, requests
import logging # Add logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 테스트용 고정 URL 및 해당 날짜 정의 (2025-05-07 용)
TARGET_NCM_DATE = dt.date(2025, 5, 7) # 사용자께서 제공한 파일명의 날짜
TARGET_NCM_URL = "https://assets.ncm.gov.ae/assets/bulletins/2025/05/2-20250507011511.pdf"

# 이전의 다른 URL 정의들은 이 테스트를 위해 주석 처리하거나 삭제합니다.
# FIXED_TEST_URL_DATE = dt.date(2025, 5, 8)
# FIXED_TEST_URL = "https://assets.ncm.gov.ae/assets/bulletins/2025/05/2-20250508000209.pdf"
# URL_FMT = ("https://www.ncm.ae/content/files/forecasts/marine/Marine_Forecast_{:%Y%m%d}.pdf")

def fetch_pdf(date: dt.date) -> bytes:
    url_to_fetch = None
    if date == TARGET_NCM_DATE:
        url_to_fetch = TARGET_NCM_URL
        logging.info(f"NCM PDF fetch: Using TARGET_NCM_URL for date {date.strftime('%Y%m%d')}: {url_to_fetch}")
    else:
        logging.info(f"NCM PDF fetch: Date {date.strftime('%Y%m%d')} is not the TARGET_NCM_DATE ({TARGET_NCM_DATE.strftime('%Y%m%d')}). Skipping NCM PDF fetch for this date.")
        return None

    if url_to_fetch:
        try:
            res = requests.get(url_to_fetch, timeout=20)
            if res.status_code == 200:
                logging.info(f"NCM PDF fetch: Successfully fetched PDF (status {res.status_code}) for date: {date.strftime('%Y%m%d')} from {url_to_fetch}")
                return res.content
            else:
                logging.warning(f"NCM PDF fetch: Failed to fetch PDF (status {res.status_code}) for date: {date.strftime('%Y%m%d')} from {url_to_fetch}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"NCM PDF fetch: RequestException for URL {url_to_fetch}: {e}")
            return None
    return None

def _max_from_range(txt: str) -> float | None:
    if pd.isna(txt) or txt is None:
        return None
    # Use findall to capture all numbers (int or float) in the string
    nums_str = re.findall(r"\\d+\\.?\\d*|\\d+", str(txt))
    if not nums_str:
        logging.debug(f"NCM Parser (_max_from_range): No numbers found in text: '{txt}'")
        return None
    nums_float = []
    for n_str in nums_str:
        try:
            nums_float.append(float(n_str))
        except ValueError:
            logging.warning(f"NCM Parser (_max_from_range): Could not convert '{n_str}' to float from text '{txt}'")
            continue
    if not nums_float:
        logging.debug(f"NCM Parser (_max_from_range): No valid numbers extracted from '{txt}'")
        return None
    # Return the maximum value found
    max_val = max(nums_float)
    logging.debug(f"NCM Parser (_max_from_range): Extracted max value {max_val} from '{txt}'")
    return max_val

def parse_ncm(date: dt.date, area="ARABIAN GULF", zone="OFF SHORE") -> pd.DataFrame:
    """
    Parses the NCM Marine Forecast PDF for a specific date to extract max wave height (ft)
    and max wind speed (kt) for the ARABIAN GULF OFF SHORE zone.

    Args:
        date: The target date for the forecast.
        area: The sea area (currently only supports ARABIAN GULF).
        zone: The specific zone (currently only supports OFF SHORE implicitly).

    Returns:
        A pandas DataFrame with columns ['date', 'wave_ft_max_off', 'wind_kn_max_off'].
        Returns None values if parsing fails or data is not found.
    """
    logging.info(f"NCM Parser (parse_ncm): Attempting to parse NCM data for date: {date.strftime('%Y%m%d')}")
    pdf_bytes = fetch_pdf(date)
    if pdf_bytes is None:
        logging.warning(f"NCM Parser (parse_ncm): fetch_pdf returned no data for date {date.strftime('%Y%m%d')}. Cannot parse.")
        # Return DataFrame with Nones as expected by downstream processing
        return pd.DataFrame({"date": [date], "wave_ft_max_off": [None], "wind_kn_max_off": [None]})

    try:
        # Increased tolerances might help with slightly misaligned tables
        tables = camelot.read_pdf(io.BytesIO(pdf_bytes), pages="1", flavor="stream", edge_tol=500, row_tol=15, strip_text='\\n\\r')
    except Exception as e:
        logging.error(f"NCM Parser (parse_ncm): Camelot failed to read PDF for date {date.strftime('%Y%m%d')}: {e}")
        return pd.DataFrame({"date": [date], "wave_ft_max_off": [None], "wind_kn_max_off": [None]})
        
    if not tables:
        logging.warning(f"NCM Parser (parse_ncm): Camelot found no tables in PDF for date {date.strftime('%Y%m%d')}.")
        return pd.DataFrame({"date": [date], "wave_ft_max_off": [None], "wind_kn_max_off": [None]})

    # Assume the relevant table is the first one found
    df_raw = tables[0].df
    logging.info(f"NCM Parser (parse_ncm): Raw table 0 head from PDF ({date.strftime('%Y%m%d')}):\n{df_raw.head().to_string()}")

    wave_val = None
    wind_val = None
    
    try:
        # --- Revised Parsing Logic based on observed table structure ---
        
        # 1. Find row indices for WAVE HEIGHT and WIND SPEED
        wave_row_idx = -1
        wind_row_idx = -1
        # Search in the first column (index 0) for the keywords
        for idx, cell_value in enumerate(df_raw.iloc[:, 0]):
            cell_str = str(cell_value).upper().strip()
            if "WAVE HEIGHT" in cell_str:
                wave_row_idx = idx
            elif "WIND SPEED" in cell_str:
                wind_row_idx = idx
            # Break early if both found
            if wave_row_idx != -1 and wind_row_idx != -1:
                break
        
        logging.info(f"NCM Parser: Found WAVE HEIGHT row at index: {wave_row_idx}")
        logging.info(f"NCM Parser: Found WIND SPEED row at index: {wind_row_idx}")

        if wave_row_idx == -1 or wind_row_idx == -1:
            logging.warning("NCM Parser: Could not find WAVE HEIGHT or WIND SPEED rows in the first column.")
            raise ValueError("Required metric rows not found")

        # 2. Find column index for the target date
        date_col_idx = -1
        target_date_str = date.strftime("%d/%m/%Y") # Format matches the PDF table
        # Search in the second row (index 1) for the date string
        if len(df_raw) > 1: # Check if the second row exists
            for idx, cell_value in enumerate(df_raw.iloc[1, :]):
                cell_str = str(cell_value).strip()
                if cell_str == target_date_str:
                    date_col_idx = idx
                    break
        
        logging.info(f"NCM Parser: Found target date '{target_date_str}' column at index: {date_col_idx}")

        if date_col_idx == -1:
            logging.warning(f"NCM Parser: Could not find column for target date '{target_date_str}' in the second row.")
            raise ValueError("Target date column not found")

        # 3. Extract values using found indices
        if len(df_raw) > max(wave_row_idx, wind_row_idx) and len(df_raw.columns) > date_col_idx:
            wave_text = df_raw.iloc[wave_row_idx, date_col_idx]
            wind_text = df_raw.iloc[wind_row_idx, date_col_idx]
            
            logging.info(f"NCM Parser: Extracted text for date {target_date_str}: Wave='{wave_text}', Wind='{wind_text}'")
            
            # 4. Convert text to max numeric value
            wave_val = _max_from_range(wave_text)
            wind_val = _max_from_range(wind_text)
        else:
             logging.warning(f"NCM Parser: Found indices ({wave_row_idx=}, {wind_row_idx=}, {date_col_idx=}) are out of bounds for the table shape {df_raw.shape}.")
             raise IndexError("Table indices out of bounds")

    except Exception as e:
        logging.error(f"NCM Parser (parse_ncm): Failed to parse table structure for date {date.strftime('%Y%m%d')}: {e}")
        # wave_val and wind_val will remain None

    # Create the final DataFrame
    parsed_data = pd.DataFrame({
        "date": [date],
        "wave_ft_max_off": [wave_val],
        "wind_kn_max_off": [wind_val],
    })
    
    if wave_val is not None or wind_val is not None:
        logging.info(f"NCM Parser (parse_ncm): Successfully parsed NCM data for {date.strftime('%Y%m%d')}: Max Wave {wave_val} ft, Max Wind {wind_val} kt")
    else:
        logging.warning(f"NCM Parser (parse_ncm): Failed to extract valid NCM values for {date.strftime('%Y%m%d')}.")
        
    return parsed_data

if __name__ == '__main__':
    logging.info("NCM Parser Test Run (__main__)")
    
    logging.info("\n--- Testing with TARGET_NCM_DATE ---")
    df_target_test = parse_ncm(TARGET_NCM_DATE)
    logging.info("\nParsed NCM Data (TARGET_NCM_DATE):")
    print(df_target_test) # Use print for final output clarity in terminal

    # Test a different date to ensure it skips fetching
    logging.info("\n--- Testing with a different date (should skip fetch) ---")
    other_date = TARGET_NCM_DATE + dt.timedelta(days=1)
    df_other_test = parse_ncm(other_date)
    logging.info("\nParsed NCM Data (Other Date):")
    print(df_other_test)

    logging.info("\n--- Testing _max_from_range function ---")
    test_ranges = {
        "2 - 3 / 5 FT": 5.0, # From PDF example
        "10 - 18 / 20 KT": 20.0, # From PDF example
        "2 - 4": 4.0, "5": 5.0, "0.5 - 1.0": 1.0, "Around 1.5": 1.5,
        "Less than 0.5": 0.5, "1-2 FT": 2.0, "text only": None,
        None: None, "": None, "10-15 KT": 15.0, "20 / 30": 30.0
    }
    all_tests_passed = True
    for text, expected in test_ranges.items():
        result = _max_from_range(text)
        if result != expected:
            logging.error(f"Test failed for _max_from_range with input: '{text}'. Got {result}, expected {expected}")
            all_tests_passed = False
        else:
            logging.info(f'_max_from_range("{text}") -> {result} (Expected: {expected})')
    if all_tests_passed:
        logging.info("_max_from_range tests completed successfully.")
    else:
        logging.error("_max_from_range tests completed with errors.") 