#!/usr/bin/env python
"""Test OpenAQ and WAQI API lookups with project cities."""
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_api import fetch_latest_by_city, compute_aqi_from_pm25, fetch_from_waqi, fetch_nearest_by_city
import config


def main():
    # Get city from command line or auto-detect from project files
    if len(sys.argv) > 1:
        city = sys.argv[1]
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [f for f in os.listdir(project_dir) if f.endswith('-air-quality.csv')]
        if candidates:
            file_base = candidates[0]
            city = os.path.splitext(file_base)[0].replace('-', ' ').replace('_', ' ')
            print(f'Using project city: {city}')
        else:
            city = 'Delhi'

    print(f'Testing lookups for: {city}\n')

    # Try OpenAQ direct city lookup
    print('1. Trying direct OpenAQ city lookup...')
    data = fetch_latest_by_city(city)
    if data:
        print('SUCCESS - OpenAQ direct city lookup returned data')
    else:
        print('No data - trying nearest OpenAQ stations...')
        
        # Try nearest by geocoding
        data = fetch_nearest_by_city(city)
        if data:
            print('SUCCESS - OpenAQ nearest stations returned data')
        else:
            print('No data - trying WAQI fallback...')
            
            # Try WAQI fallback
            token = config.get_waqi_token()
            if token:
                data = fetch_from_waqi(city, token)
                if data:
                    print('SUCCESS - WAQI fallback returned data')
                else:
                    print('No data from WAQI')
            else:
                print('No WAQI token configured')

    if data:
        print('\nPollutant snapshot:')
        for k, v in data.items():
            if v is not None:
                print(f'  {k}: {v}')
        
        aqi = compute_aqi_from_pm25(data.get('pm25'))
        if aqi:
            print(f'\nEstimated AQI (from PM2.5): {aqi}')
    else:
        print('\nNo data available from any source')


if __name__ == '__main__':
    main()
