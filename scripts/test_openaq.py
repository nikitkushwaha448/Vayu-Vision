"""Manual test script for OpenAQ client.
Run: python scripts/test_openaq.py "City Name"
"""
import sys
from realtime_api import fetch_latest_by_city, compute_aqi_from_pm25, fetch_from_waqi
import config


def main():
    # If a city is provided on the command line use it. Otherwise attempt to pick
    # a city from local project CSV files to match what you've added to the project.
    import os
    project_dir = os.path.dirname(os.path.dirname(__file__))

    if len(sys.argv) > 1:
        city = sys.argv[1]
    else:
        # find first '*-air-quality.csv' file in project
        candidates = [f for f in os.listdir(project_dir) if f.endswith('-air-quality.csv')]
        if candidates:
            # derive a display name from the filename
            file_base = candidates[0]
            city = os.path.splitext(file_base)[0].replace('-', ' ').replace('_', ' ')
            print('No city arg provided - using project file-derived city:', city)
        else:
            city = 'Delhi'
    data = fetch_latest_by_city(city)
    if not data:
        print('No direct OpenAQ city data for', city)
        # Try nearest station lookup via geocoding
        print('Trying nearest OpenAQ stations by geocoding city...')
        from realtime_api import fetch_nearest_by_city
        data = fetch_nearest_by_city(city)

        if not data:
            token = config.get_waqi_token()
            if token:
                print('Trying WAQI fallback using provided token...')
                data = fetch_from_waqi(city, token)
        if not data:
            print('No data available from OpenAQ or WAQI for', city)
            return

    print('OpenAQ pollutant snapshot for', city)
    for k, v in data.items():
        print(f'  {k}: {v}')

    est = compute_aqi_from_pm25(data.get('pm25'))
    print('Estimated AQI from PM2.5:', est)


if __name__ == '__main__':
    main()
