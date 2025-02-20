import requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests

# Constants
GOOGLE_API_KEY = 'AIzaSyBXP0aw9LB7Tql656l63wXmOnFC0z-MAt4'
CSV_FILE_PATH = 'Data/test_data2.csv'
OUTPUT_FILE_PATH = 'output_with_coordinates.csv'
WEATHER_OUTPUT_FILE_PATH = 'weather_data.csv'
MONTHLY_OUTPUT_FILE_PATH = 'monthly_aggregated_data.csv'


def get_coordinates(address, api_key):
    """
    Get coordinates from Google Maps API
    """
    params = {'key': api_key, 'address': address}
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
    print("Requesting:", base_url, params)
    response = requests.get(base_url, params=params).json()

    if response['status'] == 'OK':
        geometry = response['results'][0]['geometry']
        return geometry['location']['lat'], geometry['location']['lng']
    else:
        return None, None


def get_weather_data(latitude, longitude, start_date, end_date):
    """
    Fetch weather data from Open-Meteo API
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum", "rain_sum", "snowfall_sum",
                  "wind_speed_10m_max", "wind_gusts_10m_max"]
    }

    responses = openmeteo.weather_api(url, params=params)

    if responses:
        response = responses[0]
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
        return pd.DataFrame(data=daily_data)
    return pd.DataFrame()


def aggregate_monthly_data(weather_df, coordinates_df, price_df=None):
    """
    Aggregate weather data by month and combine with price and coordinate data
    """
    # Convert date column to datetime
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Extract month and year from date
    weather_df['month_year'] = weather_df['date'].dt.to_period('M')

    # Group by month_year and location, calculate averages
    monthly_weather = weather_df.groupby(['month_year', 'FullAddress']).agg({
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'mean',
        'rain_sum': 'mean',
        'snowfall_sum': 'mean',
        'wind_speed_10m_max': 'mean',
        'wind_gusts_10m_max': 'mean'
    }).reset_index()

    # Merge with coordinates
    result = monthly_weather.merge(coordinates_df, on='FullAddress', how='left')

    # If price data is provided, merge it
    if price_df is not None:
        if 'date' in price_df.columns:
            price_df['month_year'] = pd.to_datetime(price_df['date']).dt.to_period('M')

        if 'price' in price_df.columns:
            result = result.merge(
                price_df[['month_year', 'FullAddress', 'price']],
                on=['month_year', 'FullAddress'],
                how='left'
            )

    # Clean up column names
    result = result.rename(columns={
        'temperature_2m_mean': 'avg_temperature',
        'precipitation_sum': 'avg_precipitation',
        'rain_sum': 'avg_rain',
        'snowfall_sum': 'avg_snowfall',
        'wind_speed_10m_max': 'avg_wind_speed',
        'wind_gusts_10m_max': 'avg_wind_gusts'
    })

    # Sort by date and location
    result = result.sort_values(['month_year', 'FullAddress'])

    return result


def main():
    # Load the Zillow dataset
    df = pd.read_csv(CSV_FILE_PATH)
    df.dropna(inplace=True)
    df['FullAddress'] = df['RegionName'].astype(str) + ', ' + df['City'] + ', ' + df['State']

    # Get coordinates for all addresses
    coordinates_output = []
    for address in df['FullAddress']:
        lat, lon = get_coordinates(address, GOOGLE_API_KEY)
        coordinates_output.append({'FullAddress': address, 'Latitude': lat, 'Longitude': lon})

    coordinates_df = pd.DataFrame(coordinates_output)
    coordinates_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Coordinates saved to {OUTPUT_FILE_PATH}")

    # Fetch weather data for each location
    weather_data_frames = []
    for _, row in coordinates_df.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            print(f"Fetching weather data for {row['FullAddress']}...")
            weather_df = get_weather_data(row['Latitude'], row['Longitude'], "2024-10-24", "2024-11-07")
            weather_df['FullAddress'] = row['FullAddress']
            weather_data_frames.append(weather_df)

    if weather_data_frames:
        weather_data = pd.concat(weather_data_frames, ignore_index=True)
        weather_data.to_csv(WEATHER_OUTPUT_FILE_PATH, index=False)
        print(f"Weather data saved to {WEATHER_OUTPUT_FILE_PATH}")

        # Create monthly aggregated dataset
        # Note: Add your price data here if available
        monthly_data = aggregate_monthly_data(weather_data, coordinates_df)
        monthly_data.to_csv(MONTHLY_OUTPUT_FILE_PATH, index=False)
        print(f"Monthly aggregated data saved to {MONTHLY_OUTPUT_FILE_PATH}")


if __name__ == '__main__':
    main()