import aiohttp
import asyncio
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests

# Constants
GOOGLE_API_KEY = 'AIzaSyC8LRtjoAFF2_sP-x4uQ_5JO2a8E98kk6E'
CSV_FILE_PATH = 'Data/zillow_data.csv'
OUTPUT_FILE_PATH = 'output_with_coordinates.csv'
WEATHER_OUTPUT_FILE_PATH = 'weather_data.csv'
MONTHLY_OUTPUT_FILE_PATH = 'monthly_aggregated_data.csv'


async def get_coordinates(session, address, api_key):
    """
    Get coordinates from Google Maps API asynchronously
    """
    params = {'key': api_key, 'address': address}
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
    print("Requesting:", base_url, params)
    async with session.get(base_url, params=params) as response:
        result = await response.json()
        print(result)

        if result['status'] == 'OK':
            geometry = result['results'][0]['geometry']
            return geometry['location']['lat'], geometry['location']['lng']
        else:
            return None, None


async def fetch_all_coordinates(addresses, api_key):
    async with aiohttp.ClientSession() as session:
        tasks = [get_coordinates(session, address, api_key) for address in addresses]
        return await asyncio.gather(*tasks)


async def main():
    # Load the Zillow dataset
    df = pd.read_csv(CSV_FILE_PATH)
    df.dropna(inplace=True)
    df['FullAddress'] = df['RegionName'].astype(str) + ', ' + df['City'] + ', ' + df['State']

    # Get coordinates for all addresses asynchronously
    print("Fetching coordinates asynchronously...")
    coordinates_output = await fetch_all_coordinates(df['FullAddress'], GOOGLE_API_KEY)

    coordinates_df = pd.DataFrame({
        'FullAddress': df['FullAddress'],
        'Latitude': [coord[0] for coord in coordinates_output],
        'Longitude': [coord[1] for coord in coordinates_output]
    })

    coordinates_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Coordinates saved to {OUTPUT_FILE_PATH}")


if __name__ == '__main__':
    asyncio.run(main())
