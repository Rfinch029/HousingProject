import requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import datetime

# Constants
# CSV_FILE_PATH = 'Data/zillow_data.csv'
CSV_FILE_PATH = 'Data/zillow_data_part(9000-12895).xlsx'
api_key = 'AIzaSyC8LRtjoAFF2_sP-x4uQ_5JO2a8E98kk6E'
OUTPUT_GPS_FILE_PATH = 'Preprocessing/Output/gps/output_with_coordinates3.csv'
OUTPUT_WEATHER_FILE_PATH = 'Preprocessing/Output/weather/weather_data.csv'
OUTPUT_FINAL_FILE_PATH = 'Preprocessing/Output/final/monthly_aggregated_data.csv'


def get_coordinates(address):
    params = {'key': api_key, 'address': address}
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
    print("Requesting:", base_url, params)
    response = requests.get(base_url, params=params).json() # uses params dictionary to request information from url

    if response['status'] == 'OK': # checks to see if status request is valid
        geometry = response['results'][0]['geometry']
        print(geometry['location']['lat'], geometry['location']['lng'])
        return geometry['location']['lat'], geometry['location']['lng']
    else:
        return None, None



def get_weather_data(coordinates_df, start_date, end_date):
    """
    Fetch weather data for all locations with a single API call

    Parameters:
    coordinates_df: DataFrame with columns 'FullAddress', 'Latitude', 'Longitude'
    start_date: Start date for weather data in YYYY-MM-DD format
    end_date: End date for weather data in YYYY-MM-DD format

    Returns:
    DataFrame with weather data for all locations
    """
    # Filter out rows with null coordinates
    valid_coordinates = coordinates_df[
        pd.notnull(coordinates_df['Latitude']) &
        pd.notnull(coordinates_df['Longitude'])
    ].copy()

    if valid_coordinates.empty:
        return pd.DataFrame()

    # Setup cache and retry functionality
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    # Create a single params dictionary with arrays for latitude and longitude
    params = {
        "latitude": valid_coordinates['Latitude'].tolist(),
        "longitude": valid_coordinates['Longitude'].tolist(),
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum", "rain_sum", "snowfall_sum",
                  "wind_speed_10m_max", "wind_gusts_10m_max"]
    }

    # Make a single API request
    responses = openmeteo.weather_api(url, params=params)

    print("API request completed, should only appear once")

    # Process all responses
    weather_data_frames = []

    for i, response in enumerate(responses):
        daily = response.Daily()

        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(1).ValuesAsNumpy(),
            "rain_sum": daily.Variables(2).ValuesAsNumpy(),
            "snowfall_sum": daily.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m_max": daily.Variables(4).ValuesAsNumpy(),
            "wind_gusts_10m_max": daily.Variables(5).ValuesAsNumpy()
        }

        # Create DataFrame from the data
        weather_df = pd.DataFrame(data=daily_data)

        # Add the location identifier
        weather_df['FullAddress'] = valid_coordinates.iloc[i]['FullAddress']

        weather_data_frames.append(weather_df)

    # Combine all results
    if weather_data_frames:
        return pd.concat(weather_data_frames, ignore_index=True)
    else:
        return pd.DataFrame()


def aggregate_monthly_data(weather_df, coordinates_df, price_df=None):
    """
    Aggregate weather data by month and combine with price and coordinate data.
    If price_df is provided, it should contain a 'date' column and a 'price' column.
    For price data, this function aggregates all prices for a house in a given month into a comma-separated string.
    """
    # Convert date column to datetime
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    # Extract month and year from date
    weather_df['month_year'] = weather_df['date'].dt.to_period('M')

    # Group weather data by month_year and location, calculating averages
    monthly_weather = weather_df.groupby(['month_year', 'FullAddress']).agg({
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'mean',
        'rain_sum': 'mean',
        'snowfall_sum': 'mean',
        'wind_speed_10m_max': 'mean',
        'wind_gusts_10m_max': 'mean'
    }).reset_index()

    # Merge with coordinates to add latitude/longitude etc.
    result = monthly_weather.merge(coordinates_df, on='FullAddress', how='left')

    # Merge with price data if provided
    if price_df is not None:
        # Ensure the price date is a datetime and create a month_year column
        if 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df['month_year'] = price_df['date'].dt.to_period('M')
        
        # Group the price data by month and address,
        # aggregating all prices into a comma-separated string rather than a list.
        if 'price' in price_df.columns:
            price_agg = price_df.groupby(['month_year', 'FullAddress'])['price'] \
                .apply(lambda x: ','.join(map(str, x))) \
                .reset_index()
            # Merge the aggregated price data with the weather data
            result = result.merge(price_agg, on=['month_year', 'FullAddress'], how='left')

    # Rename columns for clarity
    result = result.rename(columns={
        'temperature_2m_mean': 'avg_temperature',
        'precipitation_sum': 'avg_precipitation',
        'rain_sum': 'avg_rain',
        'snowfall_sum': 'avg_snowfall',
        'wind_speed_10m_max': 'avg_wind_speed',
        'wind_gusts_10m_max': 'avg_wind_gusts'
    })

    # Sort by month_year and FullAddress for clarity
    result = result.sort_values(['month_year', 'FullAddress'])
     # Remove the FullAddress column from the final result
    result = result.drop(columns=['FullAddress'])

    return result

def main():
    # Define start and end dates as strings
    start_date_str = "1996-04-01"
    end_date_str = "1998-04-01"
    
    # Convert to datetime objects and break into components
    start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    start_year, start_month, start_day = start_date_dt.year, start_date_dt.month, start_date_dt.day
    end_year, end_month, end_day = end_date_dt.year, end_date_dt.month, end_date_dt.day
    
    print("Start Date Components:", start_year, start_month, start_day)
    print("End Date Components:", end_year, end_month, end_day)
    
    '''
    -------------NO NEED, ALREADY HAVE COORDINATE CSV FILE----------------------------------
    
    # Load the Zillow dataset which contains address and price data
    df = pd.read_excel(CSV_FILE_PATH)
    # df.dropna(inplace=True)
    
    # Build a full address string to be used for geocoding
    df['FullAddress'] = df['RegionName'].astype(str) + ', ' + df['City'] + ', ' + df['State']

    # --- Reshape price columns ---
    # Assuming the first 7 columns are non-price data,
    # and that price data (with date headers) starts at column 8.
    non_price_columns = df.columns[:7].tolist()
    if 'FullAddress' not in non_price_columns:
        non_price_columns.append('FullAddress')
    
    # Melt the DataFrame so that the price columns become rows.
    # The resulting DataFrame will have a 'date' column (from the former header names)
    # and a 'price' column (the corresponding values).
    price_df = pd.melt(df, id_vars=non_price_columns, var_name="date", value_name="price")
    
    # Convert the melted 'date' column to datetime and filter by our date range
    price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
    price_df = price_df[(price_df['date'] >= start_date_dt) & (price_df['date'] <= end_date_dt)]
    # Convert back to string in the format YYYY-MM-DD if needed
    price_df['date'] = price_df['date'].dt.strftime('%Y-%m-%d')
    
    # Get coordinates for all addresses
    coordinates_output = []
    for address in df['FullAddress']:
        lat, lon = get_coordinates(address)
        coordinates_output.append({'FullAddress': address, 'Latitude': lat, 'Longitude': lon})
    coordinates_df = pd.DataFrame(coordinates_output)
    coordinates_df.to_csv(OUTPUT_GPS_FILE_PATH, index=False)
    print(f"Coordinates saved to {OUTPUT_GPS_FILE_PATH}")
    
    -----------------------------------------------------------------------------------------------
    '''
    
    coordinates_df = pd.read_csv(OUTPUT_GPS_FILE_PATH)
    '''
    --------------------WEATHER API BUGGIN, WILL TRY TMW WHEN TIMEOUT IS OVER------------------------
    
    # Fetch weather data using the start and end dates
    print("Fetching weather data for all locations with a single API call...")
    weather_data = get_weather_data(coordinates_df, start_date_str, end_date_str)
    
    if not weather_data.empty:
        weather_data.to_csv(OUTPUT_WEATHER_FILE_PATH, index=False)
        print(f"Weather data saved to {OUTPUT_WEATHER_FILE_PATH}")
        
        # Create monthly aggregated dataset and merge with the melted price data
        monthly_data = aggregate_monthly_data(weather_data, coordinates_df, price_df=price_df)
        monthly_data.to_csv(OUTPUT_FINAL_FILE_PATH, index=False)
        print(f"Monthly aggregated data saved to {OUTPUT_FINAL_FILE_PATH}")
    else:
        print("No weather data was retrieved.")

    -------------------------------------------------------------------------------------------------
    '''

if __name__ == '__main__':
    main()

