import requests
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


# Constants
GOOGLE_API_KEY = 'AIzaSyBXP0aw9LB7Tql656l63wXmOnFC0z-MAt4'
CSV_FILE_PATH = 'Data/test_data2.csv'
OUTPUT_FILE_PATH = 'output_with_coordinates.csv'
WEATHER_OUTPUT_FILE_PATH = 'weather_data.csv'


# Function to get coordinates from Google Maps API
def get_coordinates(address, api_key):
   params = {'key': api_key, 'address': address}
   base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
   print("Requesting:", base_url, params)
   response = requests.get(base_url, params=params).json()


   if response['status'] == 'OK':
       geometry = response['results'][0]['geometry']
       return geometry['location']['lat'], geometry['location']['lng']
   else:
       return None, None


# Function to fetch weather data from Open-Meteo API
def get_weather_data(latitude, longitude, start_date, end_date):
   # Setup Open-Meteo client with caching and retries
   cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
   retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
   openmeteo = openmeteo_requests.Client(session=retry_session)


   # Define weather parameters
   url = "https://archive-api.open-meteo.com/v1/archive"
   params = {
       "latitude": latitude,
       "longitude": longitude,
       "start_date": start_date,
       "end_date": end_date,
       "daily": ["temperature_2m_mean", "precipitation_sum", "rain_sum", "snowfall_sum", "wind_speed_10m_max", "wind_gusts_10m_max"]
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


# Main function
def main():
   # Load the Zillow dataset
   df = pd.read_csv(CSV_FILE_PATH)
   df.dropna(inplace=True) # Removes rows that contain any missing values
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


if __name__ == '__main__':
   main()
