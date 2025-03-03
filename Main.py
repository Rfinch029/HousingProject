import aiohttp
import asyncio
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

# Constants
CSV_FILE_PATH = 'Data/zillow_data.csv'
OUTPUT_FILE_PATH = 'output_with_coordinates.csv'
WEATHER_OUTPUT_FILE_PATH = 'weather_data.csv'
MONTHLY_OUTPUT_FILE_PATH = 'monthly_aggregated_data.csv'


async def get_coordinates(session, address):
    """
    Get coordinates from OpenStreetMap Nominatim API asynchronously
    """
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    base_url = 'https://nominatim.openstreetmap.org/search'

    headers = {
        'User-Agent': 'PropertyWeatherAnalysis/1.0'  # Nominatim requires a user agent
    }

    try:
        async with session.get(base_url, params=params, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                if result and len(result) > 0:
                    return float(result[0]['lat']), float(result[0]['lon'])
                else:
                    print(f"No results found for address: {address}")
                    return None, None
            else:
                print(f"Error {response.status} for address: {address}")
                return None, None
    except Exception as e:
        print(f"Exception for {address}: {str(e)}")
        return None, None


async def fetch_all_coordinates(addresses):
    # Add a delay between requests to respect Nominatim's usage policy
    # (they recommend 1 request per second)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for address in addresses:
            # Add each task with a small delay
            tasks.append(get_coordinates(session, address))
            await asyncio.sleep(1.1)  # Slightly more than 1 second to be safe

        return await asyncio.gather(*tasks)


async def main():
    # Load the Zillow dataset
    df = pd.read_csv(CSV_FILE_PATH)
    df.dropna(inplace=True)
    df['FullAddress'] = df['RegionName'].astype(str) + ', ' + df['City'] + ', ' + df['State']

    # Get coordinates for all addresses asynchronously
    print("Fetching coordinates asynchronously...")
    coordinates_output = await fetch_all_coordinates(df['FullAddress'])

    coordinates_df = pd.DataFrame({
        'FullAddress': df['FullAddress'],
        'Latitude': [coord[0] for coord in coordinates_output],
        'Longitude': [coord[1] for coord in coordinates_output]
    })

    coordinates_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Coordinates saved to {OUTPUT_FILE_PATH}")


if __name__ == '__main__':
    asyncio.run(main())