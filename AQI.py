import pandas as pd
import joblib
import realtime_api


def aqi_checker(city_name):
    pollutant_values = realtime_api.fetch_latest_by_city(city_name)
    if not pollutant_values:
        print('No OpenAQ data available for', city_name)
        return

    pm25 = pollutant_values.get('pm25')
    pm10 = pollutant_values.get('pm10')
    o3 = pollutant_values.get('o3')
    no2 = pollutant_values.get('no2')
    so2 = pollutant_values.get('so2')
    co = pollutant_values.get('co')

    print('Pollutants for', city_name)
    print('PM2.5:', pm25)
    print('PM10:', pm10)
    print('O3:', o3)
    print('NO2:', no2)
    print('SO2:', so2)
    print('CO:', co)

    # Try to load a local model (if exists) and predict; otherwise estimate AQI from PM2.5
    try:
        rf_model = joblib.load('Faridabad_random_forest_model.pkl')
        imputer = joblib.load('imputer.pkl')
        input_data = {
            'pm25': [pm25 or 0.0],
            'pm10': [pm10 or 0.0],
            'o3': [o3 or 0.0],
            'no2': [no2 or 0.0],
            'so2': [so2 or 0.0],
            'co': [co or 0.0],
        }
        input_df = pd.DataFrame(input_data)
        input_imputed = imputer.transform(input_df)
        rf_aqi = rf_model.predict(input_imputed)
        print('Predicted AQI (Random Forest) for', city_name, 'is:', rf_aqi[0])
    except Exception:
        est = realtime_api.compute_aqi_from_pm25(pm25)
        print('Estimated AQI (from PM2.5) for', city_name, 'is:', est)


if __name__ == '__main__':
    aqi_checker('DITE Okhla, Delhi, Delhi, India')





