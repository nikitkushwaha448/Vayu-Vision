import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import pickle
import base64
import json
import matplotlib.pyplot as plt
from pathlib import Path

from analysis import (
    process_pollutant,
    monthly_analysis,
    monthly_trend_analysis,
)

BASE_DIR = Path(__file__).resolve().parent
NO_LOCAL_MODEL_MESSAGE = "No local trained model available for this city yet, showing live AQI from WAQI."


def local_file(name):
    return BASE_DIR / name


def load_city_history(city_name):
    city_file = city_data_map.get(city_name)
    if not city_file:
        return None

    city_file_path = local_file(city_file)
    if not city_file_path.exists():
        return None

    df = pd.read_csv(city_file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    numeric_columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'AQI']
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df


def get_aqi_status(aqi_value):
    if aqi_value <= 50:
        return 'Good', 'green'
    if aqi_value <= 100:
        return 'Moderate', '#d4b000'
    if aqi_value <= 150:
        return 'Unhealthy for Sensitive Groups', '#d97706'
    if aqi_value <= 200:
        return 'Unhealthy', 'red'
    if aqi_value <= 300:
        return 'Very Unhealthy', 'orange'
    return 'Hazardous', '#6b21a8'


def get_health_actions(aqi_value):
    if aqi_value <= 50:
        return [
            'Outdoor activity is generally safe.',
            'Keep ventilation normal and maintain routine monitoring.',
            'Sensitive individuals can continue normal activity.'
        ]
    if aqi_value <= 100:
        return [
            'Reduce very long outdoor exertion if irritation starts.',
            'Keep windows open during cleaner hours when possible.',
            'Sensitive groups should monitor breathing discomfort.'
        ]
    if aqi_value <= 150:
        return [
            'Sensitive groups should limit prolonged outdoor exposure.',
            'Prefer indoor exercise or lower-intensity activity.',
            'Use a mask outdoors if exposure cannot be avoided.'
        ]
    if aqi_value <= 200:
        return [
            'Limit outdoor activity for everyone, especially children and older adults.',
            'Keep indoor air cleaner using filtration or closed windows during peak pollution.',
            'Use protective masks outdoors and avoid traffic-heavy areas.'
        ]
    if aqi_value <= 300:
        return [
            'Avoid outdoor exercise and keep exposure short.',
            'Sensitive groups should stay indoors as much as possible.',
            'Use air purifiers if available and seek medical advice for persistent symptoms.'
        ]
    return [
        'Stay indoors except for essential travel.',
        'Use sealed indoor spaces with filtration if available.',
        'Seek medical help promptly if breathing distress, chest tightness, or severe irritation occurs.'
    ]


def calculate_personal_aqi_risk(aqi_value, outdoor_hours, sensitivity_points, mask_type, has_purifier):
    mask_reduction = {
        'None': 0,
        'Cloth': 4,
        'Surgical': 8,
        'N95/FFP2': 16,
    }

    base_score = (min(float(aqi_value), 350.0) / 350.0) * 72.0
    exposure_score = min(float(outdoor_hours), 12.0) * 2.2
    sensitivity_score = float(sensitivity_points) * 4.0
    protection_bonus = mask_reduction.get(mask_type, 0) + (8 if has_purifier else 0)

    score = max(0.0, min(100.0, base_score + exposure_score + sensitivity_score - protection_bonus))

    if score <= 25:
        level = 'Low'
    elif score <= 50:
        level = 'Moderate'
    elif score <= 75:
        level = 'High'
    else:
        level = 'Very High'

    return round(score, 1), level


def build_personal_protection_actions(aqi_value, risk_level, outdoor_hours, mask_type, has_purifier):
    actions = []

    if aqi_value <= 100:
        actions.append('Plan outdoor tasks in shorter blocks and hydrate regularly.')
    elif aqi_value <= 200:
        actions.append('Reduce outdoor duration and avoid heavy activity near peak traffic hours.')
    else:
        actions.append('Avoid prolonged outdoor exposure and postpone intense activity where possible.')

    if outdoor_hours >= 3:
        actions.append('Split outdoor time into smaller sessions and use indoor breaks between activities.')

    if mask_type == 'None':
        actions.append('Use at least a surgical mask outdoors; prefer N95/FFP2 on poor AQI days.')
    elif mask_type in ['Cloth', 'Surgical'] and aqi_value > 150:
        actions.append('Upgrade to N95/FFP2 for stronger particulate protection during high AQI periods.')

    if not has_purifier and aqi_value > 150:
        actions.append('Create a cleaner indoor room: close windows during peak pollution and use filtration if available.')
    elif has_purifier:
        actions.append('Run purifier continuously during peak hours and replace filters as scheduled.')

    if risk_level in ['High', 'Very High']:
        actions.append('Keep quick-relief medicines and emergency contacts ready if you have respiratory or cardiac history.')

    return actions


def evaluate_symptom_urgency(aqi_value, selected_symptoms):
    severe_markers = {'Shortness of breath', 'Chest pain', 'Severe wheezing', 'Faintness'}
    moderate_markers = {'Persistent cough', 'Eye irritation', 'Headache', 'Throat irritation'}

    symptom_set = set(selected_symptoms)
    has_severe = len(symptom_set.intersection(severe_markers)) > 0
    has_moderate = len(symptom_set.intersection(moderate_markers)) > 0

    if has_severe:
        return 'urgent', 'Severe symptoms detected. Seek medical care immediately, especially with current AQI conditions.'
    if has_moderate and aqi_value > 150:
        return 'high', 'Symptoms plus elevated AQI detected. Minimize exposure and consult a clinician if symptoms persist.'
    if has_moderate:
        return 'medium', 'Mild-to-moderate symptoms detected. Reduce exposure, hydrate, and monitor progression.'
    return 'low', 'No major symptoms selected. Continue preventive steps and monitor for changes.'


def build_safe_outdoor_time_plan(aqi_value, personal_risk_level):
    base_minutes = {
        'Morning (6-9 AM)': 90,
        'Midday (10 AM-3 PM)': 45,
        'Evening (4-8 PM)': 70,
        'Night (after 8 PM)': 80,
    }

    if aqi_value <= 100:
        aqi_factor = 1.0
    elif aqi_value <= 150:
        aqi_factor = 0.75
    elif aqi_value <= 200:
        aqi_factor = 0.55
    elif aqi_value <= 300:
        aqi_factor = 0.35
    else:
        aqi_factor = 0.2

    risk_factor = {
        'Low': 1.0,
        'Moderate': 0.85,
        'High': 0.65,
        'Very High': 0.45,
    }.get(personal_risk_level, 0.85)

    rows = []
    for window, minutes in base_minutes.items():
        allowed_minutes = int(round(minutes * aqi_factor * risk_factor))
        if allowed_minutes >= 60:
            guidance = 'Preferred window'
        elif allowed_minutes >= 30:
            guidance = 'Limit exposure and use mask'
        elif allowed_minutes >= 10:
            guidance = 'Short essential exposure only'
        else:
            guidance = 'Avoid outdoor exposure'

        rows.append(
            {
                'time_window': window,
                'recommended_minutes': max(0, allowed_minutes),
                'guidance': guidance,
            }
        )

    return pd.DataFrame(rows)


def build_commute_safety_plan(aqi_value, personal_risk_level, group_mode, commute_minutes, travel_mode):
    if aqi_value <= 100:
        aqi_factor = 1.0
    elif aqi_value <= 150:
        aqi_factor = 0.8
    elif aqi_value <= 200:
        aqi_factor = 0.6
    elif aqi_value <= 300:
        aqi_factor = 0.4
    else:
        aqi_factor = 0.25

    risk_factor = {
        'Low': 1.0,
        'Moderate': 0.85,
        'High': 0.7,
        'Very High': 0.55,
    }.get(personal_risk_level, 0.85)

    mode_factor = {
        'Walking': 1.25,
        'Two-wheeler': 1.15,
        'Public Transport': 1.0,
        'Car (windows closed)': 0.75,
        'School Bus': 0.85,
    }.get(travel_mode, 1.0)

    commute_load = round(float(aqi_value) * (commute_minutes / 60.0) * mode_factor, 1)
    tolerated_load = 120.0 * aqi_factor * risk_factor

    if commute_load <= tolerated_load * 0.8:
        commute_status = 'Safer'
    elif commute_load <= tolerated_load:
        commute_status = 'Caution'
    else:
        commute_status = 'High Risk'

    if group_mode == 'School':
        key_actions = [
            'Prefer school bus or closed-vehicle commute over walking near heavy roads.',
            'Pack a child-size mask and ensure proper fit before departure.',
            'Schedule outdoor sports only in lower AQI windows when possible.',
        ]
    elif group_mode == 'Office':
        key_actions = [
            'Shift commute outside peak traffic times where possible.',
            'Prefer closed cabin transport and avoid open-road exposure during delays.',
            'Use indoor breaks before and after commute to reduce cumulative exposure.',
        ]
    else:
        key_actions = [
            'Use shortest low-traffic route to reduce exposure duration.',
            'Use N95/FFP2 on high AQI days and avoid exertion during travel.',
            'Plan essential trips during safer windows from the time plan.',
        ]

    route_windows = pd.DataFrame(
        [
            {'window': '6-8 AM', 'score': round(70 * aqi_factor * risk_factor, 1)},
            {'window': '8-10 AM', 'score': round(45 * aqi_factor * risk_factor, 1)},
            {'window': '5-7 PM', 'score': round(40 * aqi_factor * risk_factor, 1)},
            {'window': '7-9 PM', 'score': round(62 * aqi_factor * risk_factor, 1)},
        ]
    )
    route_windows['advice'] = route_windows['score'].apply(
        lambda x: 'Better option' if x >= 55 else ('Use mask + short route' if x >= 30 else 'Avoid if possible')
    )

    return {
        'commute_load': commute_load,
        'tolerated_load': round(tolerated_load, 1),
        'commute_status': commute_status,
        'key_actions': key_actions,
        'route_windows': route_windows,
    }


def build_snapshot_dataframe(city_name, aqi_value, pollutant_values, source_label):
    return pd.DataFrame([
        {
            'city': city_name,
            'aqi': round(float(aqi_value), 2),
            'pm25': pollutant_values['pm25'],
            'pm10': pollutant_values['pm10'],
            'o3': pollutant_values['o3'],
            'no2': pollutant_values['no2'],
            'so2': pollutant_values['so2'],
            'co': pollutant_values['co'],
            'source': source_label,
        }
    ])


def render_analysis_overview(df):
    analysis_df = df.dropna(subset=['AQI']).copy()
    if analysis_df.empty:
        st.info('No usable AQI history is available for overview metrics.')
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Records', len(analysis_df))
    c2.metric('Mean AQI', f"{analysis_df['AQI'].mean():.1f}")
    c3.metric('Peak AQI', f"{analysis_df['AQI'].max():.1f}")

    if 'date' in analysis_df.columns and analysis_df['date'].notna().any():
        date_min = analysis_df['date'].min().date()
        date_max = analysis_df['date'].max().date()
        c4.metric('Date Range', f"{date_min} to {date_max}")
    else:
        c4.metric('Date Range', 'N/A')

    top_months = analysis_df[['date', 'AQI']].dropna().copy()
    available_years = sorted(top_months['date'].dt.year.dropna().astype(int).unique().tolist())
    year_window = list(range(2013, 2027))
    selectable_years = [year for year in year_window if year in available_years]
    default_years = selectable_years.copy()

    selected_top_years = st.multiselect(
        'Select years for Top AQI Months',
        options=selectable_years,
        default=default_years,
        key='top_aqi_months_year_filter',
    )

    if selected_top_years:
        top_months = top_months[top_months['date'].dt.year.isin(selected_top_years)]
    else:
        top_months = top_months.iloc[0:0]

    top_months = top_months.sort_values('AQI', ascending=False).head(5).copy()
    selected_years_label = ', '.join(str(year) for year in selected_top_years) if selected_top_years else 'No year selected'
    st.markdown(f'### Top AQI Months ({selected_years_label})')
    if not top_months.empty:
        top_months_display = top_months.copy()
        top_months_display['date'] = top_months_display['date'].dt.strftime('%Y-%m')
        top_months_display = top_months_display.rename(columns={'date': 'month', 'AQI': 'aqi'})

        top_col1, top_col2 = st.columns(2)
        with top_col1:
            st.dataframe(top_months_display, use_container_width=True)
        with top_col2:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.bar(top_months_display['month'], top_months_display['aqi'], color='#e76f51')
            ax.set_xlabel('Month')
            ax.set_ylabel('AQI')
            ax.set_title('Highest AQI Months')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)

        st.download_button(
            'Download Top AQI Months (CSV)',
            data=top_months_display.to_csv(index=False),
            file_name='top_aqi_months_selected_years.csv',
            mime='text/csv',
        )
    else:
        st.info('No Top AQI month records available for the selected year(s).')

    corr_columns = [column for column in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'AQI'] if column in analysis_df.columns]
    corr_df = analysis_df[corr_columns].dropna()
    if len(corr_df) >= 2:
        st.markdown('### Pollutant Correlation Matrix')
        corr_matrix = corr_df.corr().round(2)
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(corr_matrix.values, cmap='YlOrRd', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.index)
        for row_index in range(len(corr_matrix.index)):
            for col_index in range(len(corr_matrix.columns)):
                ax.text(col_index, row_index, corr_matrix.iloc[row_index, col_index], ha='center', va='center', color='black', fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander('Summary Statistics'):
            st.dataframe(corr_df.describe().transpose(), use_container_width=True)


def render_alert_banner(aqi_value):
    status_text, color = get_aqi_status(aqi_value)
    actions = get_health_actions(aqi_value)

    if aqi_value <= 100:
        severity = 'Advisory'
    elif aqi_value <= 200:
        severity = 'Health Alert'
    elif aqi_value <= 300:
        severity = 'Serious Alert'
    else:
        severity = 'Emergency Alert'

    st.markdown(
        f'<div style="background: rgba(8,14,26,0.78); border:1px solid rgba(255,255,255,0.18); border-left: 6px solid {color}; border-radius:14px; padding:0.9rem 1rem; color:#eef6ff; margin:0.8rem 0;">'
        f'<div style="font-weight:700; font-size:1rem; margin-bottom:0.3rem;">{severity}: {status_text}</div>'
        f'<div style="font-size:0.92rem; color:#d8e6f8;">{actions[0]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def build_city_comparison_dataframe(pollutant):
    rows = []
    for city_name, city_file in city_data_map.items():
        city_file_path = local_file(city_file)
        if not city_file_path.exists():
            continue

        df = load_city_history(city_name)
        if df is None or pollutant not in df.columns:
            continue

        series = df[pollutant].dropna()
        if series.empty:
            continue

        latest_value = float(series.iloc[-1])
        mean_value = float(series.mean())
        rows.append(
            {
                'city': city_name,
                'latest': latest_value,
                'mean': mean_value,
            }
        )

    if not rows:
        return pd.DataFrame(columns=['city', 'latest', 'mean'])

    comparison_df = pd.DataFrame(rows).sort_values('mean', ascending=False).reset_index(drop=True)
    return comparison_df


def render_city_comparison(pollutant):
    comparison_df = build_city_comparison_dataframe(pollutant)
    if comparison_df.empty:
        st.info('No comparison data is available for this pollutant across mapped cities.')
        return

    st.markdown(f'### City Comparison for {pollutant.upper()}')
    st.dataframe(comparison_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(comparison_df['city'], comparison_df['mean'], color='#2a9d8f', label='Historical mean')
    ax.scatter(comparison_df['city'], comparison_df['latest'], color='#e76f51', label='Latest recorded', zorder=3)
    ax.set_ylabel(pollutant.upper())
    ax.set_title(f'{pollutant.upper()} comparison across cities')
    ax.tick_params(axis='x', rotation=60)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def build_all_cities_top_aqi_months(start_year=2013, end_year=2026, top_n=12):
    rows = []
    for city_name in city_data_map:
        city_df = load_city_history(city_name)
        if city_df is None or 'date' not in city_df.columns or 'AQI' not in city_df.columns:
            continue

        valid_df = city_df.dropna(subset=['date', 'AQI']).copy()
        if valid_df.empty:
            continue

        valid_df = valid_df[valid_df['date'].dt.year.between(start_year, end_year)]
        if valid_df.empty:
            continue

        monthly_df = (
            valid_df.groupby(pd.Grouper(key='date', freq='MS'))['AQI']
            .mean()
            .dropna()
            .reset_index()
        )
        if monthly_df.empty:
            continue

        monthly_df['city'] = city_name
        rows.append(monthly_df)

    if not rows:
        return pd.DataFrame(columns=['city', 'date', 'AQI'])

    merged_df = pd.concat(rows, ignore_index=True)
    merged_df = merged_df.sort_values('AQI', ascending=False).head(top_n).reset_index(drop=True)
    return merged_df


def render_all_cities_top_aqi_months(start_year=2013, end_year=2026, top_n=12):
    st.markdown(f'### Top AQI Months Across All Cities ({start_year}-{end_year})')
    top_df = build_all_cities_top_aqi_months(start_year=start_year, end_year=end_year, top_n=top_n)
    if top_df.empty:
        st.info(f'No AQI records are available across mapped cities for {start_year}-{end_year}.')
        return

    display_df = top_df.copy()
    display_df['month'] = display_df['date'].dt.strftime('%Y-%m')
    display_df['aqi'] = display_df['AQI'].round(2)
    display_df = display_df[['city', 'month', 'aqi']]

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(display_df, use_container_width=True)
    with c2:
        chart_labels = (display_df['city'] + ' | ' + display_df['month']).tolist()
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.barh(chart_labels[::-1], display_df['aqi'].tolist()[::-1], color='#ef8354')
        ax.set_xlabel('AQI')
        ax.set_title('Highest AQI Months (All Cities)')
        ax.grid(axis='x', alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)

    st.download_button(
        'Download All-Cities Top AQI Months (CSV)',
        data=display_df.to_csv(index=False),
        file_name='all_cities_top_aqi_months_2013_2026.csv',
        mime='text/csv',
    )


def build_city_leaderboard():
    rows = []
    for city_name in city_data_map:
        df = load_city_history(city_name)
        if df is None or 'AQI' not in df.columns:
            continue

        valid_df = df.dropna(subset=['AQI']).copy()
        if valid_df.empty:
            continue

        latest_aqi = float(valid_df['AQI'].iloc[-1])
        mean_aqi = float(valid_df['AQI'].mean())
        peak_aqi = float(valid_df['AQI'].max())
        rows.append(
            {
                'city': city_name,
                'latest_aqi': round(latest_aqi, 2),
                'mean_aqi': round(mean_aqi, 2),
                'peak_aqi': round(peak_aqi, 2),
            }
        )

    if not rows:
        return pd.DataFrame(columns=['city', 'latest_aqi', 'mean_aqi', 'peak_aqi'])

    return pd.DataFrame(rows).sort_values('mean_aqi', ascending=False).reset_index(drop=True)


def build_prediction_report(city_name, aqi_value, status_text, source_label, pollutant_values, overall_prediction=None, general_prediction=None, vulnerable_prediction=None):
    report = {
        'city': city_name,
        'aqi': round(float(aqi_value), 2),
        'status': status_text,
        'source': source_label,
        'pollutants': pollutant_values,
        'recommended_actions': get_health_actions(float(aqi_value)),
    }

    if overall_prediction is not None:
        report['health_prediction'] = {
            'overall': overall_prediction,
            'general_population': general_prediction,
            'vulnerable_population': vulnerable_prediction,
        }

    return report


def forecast_monthly_series(df, pollutant, months_ahead=6, target_end=None):
    if df is None or pollutant not in df.columns or 'date' not in df.columns:
        return pd.DataFrame(columns=['date', 'forecast'])

    forecast_df = df[['date', pollutant]].dropna().copy()
    if forecast_df.empty:
        return pd.DataFrame(columns=['date', 'forecast'])

    monthly_series = (
        forecast_df.groupby(pd.Grouper(key='date', freq='MS'))[pollutant]
        .mean()
        .dropna()
        .sort_index()
    )
    if monthly_series.empty:
        return pd.DataFrame(columns=['date', 'forecast'])

    history_df = monthly_series.reset_index(name='value')
    history_df['year'] = history_df['date'].dt.year
    history_df['month'] = history_df['date'].dt.month

    last_date = monthly_series.index.max()
    rows = []
    current = last_date + pd.offsets.MonthBegin(1)
    if target_end is not None:
        target_end = pd.Timestamp(target_end).to_period('M').to_timestamp()
        while current <= target_end:
            month_hist = history_df[history_df['month'] == current.month]
            if len(month_hist) >= 2:
                x = month_hist['year'].to_numpy(dtype=float)
                y_vals = month_hist['value'].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y_vals, 1)
                pred = float(intercept + slope * current.year)
            elif len(month_hist) == 1:
                pred = float(month_hist['value'].iloc[0])
            else:
                pred = float(history_df['value'].mean())

            rows.append({'date': current, 'forecast': round(pred, 2)})
            current = current + pd.offsets.MonthBegin(1)
    else:
        for _ in range(months_ahead):
            month_hist = history_df[history_df['month'] == current.month]
            if len(month_hist) >= 2:
                x = month_hist['year'].to_numpy(dtype=float)
                y_vals = month_hist['value'].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y_vals, 1)
                pred = float(intercept + slope * current.year)
            elif len(month_hist) == 1:
                pred = float(month_hist['value'].iloc[0])
            else:
                pred = float(history_df['value'].mean())

            rows.append({'date': current, 'forecast': round(pred, 2)})
            current = current + pd.offsets.MonthBegin(1)

    return pd.DataFrame(rows)


def render_forecast_panel(city_name, pollutant='AQI', months_ahead=6):
    history_df = load_city_history(city_name)
    if pollutant.upper() == 'AQI':
        current_month = pd.Timestamp.today().to_period('M').to_timestamp()
        forecast_df = forecast_monthly_series(history_df, pollutant, target_end=current_month)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], errors='coerce')
        forecast_df = forecast_df.dropna(subset=['date'])
        forecast_df = forecast_df[forecast_df['date'].dt.year >= 2025].tail(6).reset_index(drop=True)
    else:
        forecast_df = forecast_monthly_series(history_df, pollutant, months_ahead=months_ahead)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], errors='coerce')
        forecast_df = forecast_df.dropna(subset=['date']).reset_index(drop=True)

    if forecast_df.empty:
        st.info(f'No forecast could be generated for {pollutant.upper()}.')
        return

    st.markdown(f'### {months_ahead}-Month {pollutant.upper()} Outlook')

    if pollutant.upper() == 'AQI':
        preview_df = forecast_df.head(3).copy()
        metric_columns = st.columns(len(preview_df))
        for index, (_, row) in enumerate(preview_df.iterrows()):
            status_text, _ = get_aqi_status(float(row['forecast']))
            label = row['date'].strftime('%Y-%m')
            metric_columns[index].metric(label, f"{row['forecast']:.1f}", status_text)

    display_df = forecast_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m')

    if pollutant.upper() == 'AQI':
        display_df['status'] = display_df['forecast'].apply(lambda value: get_aqi_status(float(value))[0])

    st.dataframe(display_df, use_container_width=True)
    st.download_button(
        f'Download {pollutant.upper()} Forecast CSV',
        data=display_df.to_csv(index=False),
        file_name=f'{pollutant.lower()}_forecast.csv',
        mime='text/csv',
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(forecast_df['date'], forecast_df['forecast'], marker='o', color='#f4a261', linewidth=2)
    ax.set_title(f'Forecast {pollutant.upper()} for {city_name}')
    ax.set_ylabel(pollutant.upper())
    ax.set_xlabel('Month')
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    if pollutant.upper() == 'AQI':
        start_label = forecast_df['date'].iloc[0].strftime('%Y-%m')
        end_label = forecast_df['date'].iloc[-1].strftime('%Y-%m')
        st.caption(f'Showing the latest 6-month AQI outlook window from {start_label} to {end_label}. 2024 AQI outlook months are excluded.')


def patch_sklearn_pickle_compat(obj):
    if obj is None:
        return obj

    estimators = getattr(obj, 'estimators_', None)
    if estimators is not None:
        for estimator in estimators:
            patch_sklearn_pickle_compat(estimator)

    if not hasattr(obj, 'monotonic_cst'):
        try:
            setattr(obj, 'monotonic_cst', None)
        except Exception:
            pass

    if hasattr(obj, 'statistics_') and not hasattr(obj, '_fill_dtype'):
        try:
            import numpy as np
            fill_dtype = obj.statistics_.dtype if getattr(obj, 'statistics_', None) is not None else np.float64
            setattr(obj, '_fill_dtype', fill_dtype)
        except Exception:
            pass

    return obj


# Load the trained Random Forest models and imputers for different cities
city_models = {
    "R.K. Puram, Delhi, Delhi, India": {
        "model": joblib.load(local_file('Delhi_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Sanjay Nagar, Ghaziabad, India": {
        "model": joblib.load(local_file('Ghaziabad_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Knowledge Park - III, Greater Noida, India": {
        "model": joblib.load(local_file('GNoida_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Talkatora, Lucknow, India": {
        "model": joblib.load(local_file('Lucknow_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Hyderabad": {
        "model": joblib.load(local_file('Hyderabad_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Mumbai": {
        "model": joblib.load(local_file('Mumbai_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Ahmedabad": {
        "model": joblib.load(local_file('Ahmedabad_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Punjab": {
        "model": joblib.load(local_file('Punjab_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Gurgaon": {
        "model": joblib.load(local_file('Gurgaon_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Chennai": {
        "model": joblib.load(local_file('Chennai_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Kerala": {
        "model": joblib.load(local_file('Kerala_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    "Nagaland": {
        "model": joblib.load(local_file('Nagaland_random_forest_model.pkl')),
        "imputer": joblib.load(local_file('imputer.pkl'))
    },
    # Add more cities as needed
}

# City display name -> API query name
city_api_map = {
    "R.K. Puram, Delhi, Delhi, India": "R.K. Puram, Delhi, Delhi, India",
    "Sanjay Nagar, Ghaziabad, India": "Sanjay Nagar, Ghaziabad, India",
    "Knowledge Park - III, Greater Noida, India": "Knowledge Park - III, Greater Noida, India",
    "Talkatora, Lucknow, India": "Talkatora, Lucknow, India",
    "Hyderabad": "Hyderabad",
    "Mumbai": "Mumbai",
    "Ahmedabad": "Ahmedabad",
    "Punjab": "Ludhiana",
    "Gurgaon": "Gurgaon",
    "Chennai": "Chennai",
    "Kerala": "Thiruvananthapuram",
    "Nagaland": "Kohima",
}

city_data_map = {
    "R.K. Puram, Delhi, Delhi, India": "r.k.-puram, delhi, delhi, india-air-quality.csv",
    "Sanjay Nagar, Ghaziabad, India": "sanjay-nagar, ghaziabad, india-air-quality.csv",
    "Knowledge Park - III, Greater Noida, India": "knowledge-park - iii, greater noida, india-air-quality.csv",
    "Talkatora, Lucknow, India": "talkatora,-lucknow, india-air-quality.csv",
    "Hyderabad": "hyderabad-air-quality.csv",
    "Mumbai": "mumbai-air-quality.csv",
    "Ahmedabad": "ahmedabad-air-quality.csv",
    "Punjab": "punjab-air-quality.csv",
    "Gurgaon": "gurgaon-air-quality.csv",
    "Chennai": "chennai-air-quality.csv",
    "Kerala": "kerala-air-quality.csv",
    "Nagaland": "nagaland-air-quality.csv",
}

city_bg_map = {
    "R.K. Puram, Delhi, Delhi, India": 'delhi.jpg',
    "Sanjay Nagar, Ghaziabad, India": 'ghaziabad.jpg',
    "Knowledge Park - III, Greater Noida, India": 'noida.jpg',
    "Talkatora, Lucknow, India": 'lucknow.jpg',
    "Hyderabad": 'hyderabad.jpg',
    "Mumbai": 'mumbai.jpg',
    "Ahmedabad": 'ahmedabad.jpg',
    "Punjab": 'punjab.jpg',
    "Gurgaon": 'gurgaon.jpg',
    "Chennai": 'chennai.jpg',
    "Kerala": 'kerala.jpg',
    "Nagaland": 'nagaland.jpg',
}

# Compatibility fix for models pickled with older scikit-learn releases.
for info in city_models.values():
    patch_sklearn_pickle_compat(info.get("model"))
    patch_sklearn_pickle_compat(info.get("imputer"))

# Replace this with your OpenAQ API key
api_key = "9122142749f2d354a43af188bc4486a59f678eed"

# Create session state variables
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

if "aqi" not in st.session_state:
    st.session_state.aqi = None

if "latest_report" not in st.session_state:
    st.session_state.latest_report = None

# Sidebar with radio buttons for page selection
page = st.sidebar.radio("Select Page", ["Home", "AQI Prediction", "Health Prediction","Analysis"])

if page == "Home":
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        css = """
        <style>
        .stApp {
            background-image: url(data:image/jpg;base64,__ENCODED_BG__);
            background-size: cover;
            background-position: center;
        }
        .vv-shell {
            background: rgba(8, 14, 26, 0.74);
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 18px;
            padding: 1.2rem 1.2rem;
            backdrop-filter: blur(4px);
            margin-bottom: 1rem;
            color: #f8fbff;
        }
        .vv-hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.4px;
            margin-bottom: 0.25rem;
            color: #ffffff;
        }
        .vv-hero-sub {
            font-size: 1.02rem;
            line-height: 1.55;
            color: #d7e8ff;
            margin-bottom: 0.8rem;
        }
        .vv-badge {
            display: inline-block;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: rgba(45, 156, 219, 0.22);
            border: 1px solid rgba(110, 193, 255, 0.45);
            color: #d8f1ff;
            font-size: 0.79rem;
            font-weight: 600;
        }
        .vv-card {
            background: rgba(6, 12, 22, 0.66);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 0.95rem;
            min-height: 160px;
            color: #f2f7ff;
        }
        .vv-card h4 {
            margin-top: 0;
            margin-bottom: 0.42rem;
            color: #e8f6ff;
            font-size: 1.03rem;
        }
        .vv-card p {
            margin: 0;
            color: #d0deef;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .vv-step {
            background: rgba(0, 0, 0, 0.36);
            border-left: 4px solid #4cb0ff;
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.55rem;
            color: #eef6ff;
            font-size: 0.93rem;
        }
        .vv-note {
            font-size: 0.88rem;
            color: #d8e6f8;
            margin-top: 0.35rem;
        }
        .vv-zone-wrap {
            margin-top: 0.7rem;
            margin-bottom: 0.35rem;
        }
        .vv-zone-bar {
            display: grid;
            grid-template-columns: repeat(6, minmax(70px, 1fr));
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.22);
        }
        .vv-zone {
            padding: 0.48rem 0.35rem;
            text-align: center;
            color: #ffffff;
            font-size: 0.74rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .vv-zone.good { background: #2a9d8f; }
        .vv-zone.mod { background: #e9c46a; color: #1f2937; }
        .vv-zone.usg { background: #f4a261; }
        .vv-zone.unh { background: #e76f51; }
        .vv-zone.vunh { background: #8d5a97; }
        .vv-zone.haz { background: #9b2226; }
        .vv-mini-note {
            margin-top: 0.45rem;
            font-size: 0.82rem;
            color: #d8e6f8;
        }
        </style>
        """
        st.markdown(
            css.replace("__ENCODED_BG__", encoded_string.decode()),
            unsafe_allow_html=True
        )


    bg_home_path = local_file('home_aqi.jpg')
    if not bg_home_path.exists():
        bg_home_path = local_file('home_bg_trend.jpg')
    if not bg_home_path.exists():
        bg_home_path = local_file('bg.jpg')
    add_bg_from_local(bg_home_path)

    st.markdown(
        """
        <div class="vv-shell">
            <div class="vv-hero-title">AQI Command Home</div>
            <div class="vv-hero-sub">
                Track city air quality risk at a glance with live AQI signals, model-backed prediction,
                and health guidance for day-to-day planning.
            </div>
            <span class="vv-badge">AQI Zones</span>
            <span class="vv-badge">Live WAQI + Local Model</span>
            <span class="vv-badge">Exposure-Aware Insights</span>
            <div class="vv-zone-wrap">
                <div class="vv-zone-bar">
                    <div class="vv-zone good">Good<br/>0-50</div>
                    <div class="vv-zone mod">Moderate<br/>51-100</div>
                    <div class="vv-zone usg">Sensitive<br/>101-150</div>
                    <div class="vv-zone unh">Unhealthy<br/>151-200</div>
                    <div class="vv-zone vunh">Very Unhealthy<br/>201-300</div>
                    <div class="vv-zone haz">Hazardous<br/>301+</div>
                </div>
                <div class="vv-mini-note">Tip: Use this zone bar as your quick daily risk reference before planning outdoor activity.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="vv-card">
                <h4>1) City AQI Check</h4>
                <p>Get current AQI and pollutant values instantly for your selected city. You get either a
                local model prediction or WAQI live fallback with category mapping.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="vv-card">
                <h4>2) Health Risk View</h4>
                <p>Translate AQI into practical health impact for overall, general, and vulnerable
                populations so actions can be prioritized quickly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="vv-card">
                <h4>3) Trend and Outlook</h4>
                <p>Inspect historical pollutant behavior and recent outlook windows to understand whether
                air quality is improving, stable, or worsening.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### System Snapshot")
    total_cities = len(city_api_map)
    model_cities = len(city_models)
    data_cities = len(city_data_map)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cities Enabled", total_cities)
    m2.metric("Model-Ready Cities", model_cities)
    m3.metric("Dataset Mapped", data_cities)
    m4.metric("Last AQI", st.session_state.aqi if st.session_state.aqi is not None else "N/A")

    st.markdown("### AQI Health Scale")
    st.markdown(
        """
        | AQI Range | Category |
        |---|---|
        | 0 - 50 | Good |
        | 51 - 100 | Moderate |
        | 101 - 150 | Unhealthy for Sensitive Groups |
        | 151 - 200 | Unhealthy |
        | 201 - 300 | Very Unhealthy |
        | 301+ | Hazardous |
        """
    )

    with st.expander("What is new in this dashboard?"):
        st.write("City-specific background themes are enabled for prediction pages.")
        st.write("Hybrid AQI pipeline supports model inference and WAQI live fallback.")
        st.write("Analysis view supports year-wise, month-wise, and monthly trend visualizations.")

    st.markdown("### Fast Workflow")
    st.markdown('<div class="vv-step"><b>Step 1:</b> Go to <b>AQI Prediction</b>, select a city, and click <b>Predict AQI</b>.</div>', unsafe_allow_html=True)
    st.markdown('<div class="vv-step"><b>Step 2:</b> Review AQI category and pollutant breakdown (PM2.5, PM10, O3, NO2, SO2, CO).</div>', unsafe_allow_html=True)
    st.markdown('<div class="vv-step"><b>Step 3:</b> Open <b>Health Prediction</b> for population-specific impact summary.</div>', unsafe_allow_html=True)
    st.markdown('<div class="vv-step"><b>Step 4:</b> Open <b>Analysis</b> to inspect historical and extended trend visuals.</div>', unsafe_allow_html=True)
    st.markdown('<div class="vv-note">Tip: If a city has no local model, the app automatically falls back to live WAQI-based display.</div>', unsafe_allow_html=True)

    trend_img_path = local_file('home_trend_card.jpg')
    if trend_img_path.exists():
        st.markdown("### Trending Signal Preview")
        st.image(str(trend_img_path), use_container_width=True)

    leaderboard_df = build_city_leaderboard()
    if not leaderboard_df.empty:
        st.markdown("### City Leaderboard")
        lead1, lead2 = st.columns(2)
        with lead1:
            st.markdown("**Most Polluted by Historical Mean**")
            st.dataframe(leaderboard_df.head(5), use_container_width=True)
        with lead2:
            st.markdown("**Cleanest by Historical Mean**")
            st.dataframe(leaderboard_df.sort_values('mean_aqi', ascending=True).head(5), use_container_width=True)


elif page == "AQI Prediction":
    st.title("Air Quality Prediction")

    st.markdown(
        """
        <style>
        .vv-panel {
            background: rgba(8, 14, 26, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.85rem;
            color: #f6fbff;
            backdrop-filter: blur(4px);
        }
        .vv-panel h4 {
            margin: 0 0 0.35rem 0;
            color: #e8f3ff;
        }
        .vv-panel p {
            margin: 0;
            color: #d4e2f0;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="vv-panel">
            <h4>Live + Model Hybrid AQI Engine</h4>
            <p>Select a city to fetch live pollutant values from WAQI. If a local model exists, AQI is
            inferred with the trained regressor; otherwise, live AQI is shown directly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dropdown menu for selecting cities
    selected_city = st.selectbox("Select city", list(city_api_map.keys()))

    # Update the background image based on the selected city

    # Add more cities as needed

    # Update the background image using CSS
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-repeat: no-repeat;
            background-size: 100% 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

    selected_bg = city_bg_map.get(selected_city, 'bg.jpg')
    bg_path = local_file(selected_bg)
    if not bg_path.exists():
        bg_path = local_file('bg.jpg')
    add_bg_from_local(bg_path)

    if selected_city not in city_models:
        st.info(NO_LOCAL_MODEL_MESSAGE)

    # Store selected city in session state
    st.session_state.selected_city = selected_city

    if st.button("Predict AQI"):
        # Fetch air quality data using OpenAQ API
        city_query = city_api_map.get(selected_city, selected_city)
        url = f"https://api.waqi.info/feed/{city_query}/?token={api_key}"
        response = requests.get(url)
        json_data = response.json()

        if json_data.get('status') == 'ok' and isinstance(json_data.get('data'), dict):
            data = json_data['data']
            aqi = data.get('aqi', 'N/A')

            # Store AQI in session state
            st.session_state.aqi = aqi

            pm25 = data['iaqi'].get('pm25', {}).get('v', 'N/A')
            pm10 = data['iaqi'].get('pm10', {}).get('v', 'N/A')
            o3 = data['iaqi'].get('o3', {}).get('v', 'N/A')
            no2 = data['iaqi'].get('no2', {}).get('v', 'N/A')
            so2 = data['iaqi'].get('so2', {}).get('v', 'N/A')
            co = data['iaqi'].get('co', {}).get('v', 'N/A')

            # Check for 'N/A' values
            if 'N/A' in [pm25, pm10, o3, no2, so2, co]:
                st.error("Some air quality parameters are not available.")
            else:
                pollutant_values = {
                    'pm25': float(pm25),
                    'pm10': float(pm10),
                    'o3': float(o3),
                    'no2': float(no2),
                    'so2': float(so2),
                    'co': float(co),
                }

                if selected_city in city_models:
                    # Create a dictionary with the retrieved air quality parameters
                    input_data = {key: [value] for key, value in pollutant_values.items()}

                    # Convert the input data to a DataFrame
                    input_df = pd.DataFrame(input_data)

                    # Impute missing values with the mean using the loaded imputer
                    imputer = city_models[selected_city]["imputer"]
                    input_imputed = imputer.transform(input_df)

                    # Predict AQI using the Random Forest model
                    rf_model = city_models[selected_city]["model"]
                    rf_aqi = rf_model.predict(input_imputed)
                    display_aqi = float(rf_aqi[0])

                    st.write(f"The AQI in {selected_city} is: {display_aqi:.2f}")

                    # Determine the air quality category and set color
                    status_text, color = get_aqi_status(display_aqi)
                    quality_message = f"<strong>{status_text.upper()}.</strong>"

                    # Combine the message and quality_message into a single string
                    styled_message = f'<p style="font-size: larger;">The air quality is <span style="color:{color}; font-size: larger;">{quality_message}</span></p>'

                    st.markdown(styled_message, unsafe_allow_html=True)
                    render_alert_banner(display_aqi)

                    status_col, source_col, avg_col = st.columns(3)
                    status_col.metric('AQI Band', status_text)
                    source_col.metric('Prediction Source', 'Local Model')

                    history_df = load_city_history(selected_city)
                    if history_df is not None and 'AQI' in history_df.columns and history_df['AQI'].notna().any():
                        historical_avg = float(history_df['AQI'].mean())
                        avg_col.metric('Vs Historical Mean', f"{display_aqi - historical_avg:+.1f}")

                        latest_row = history_df.sort_values('date').dropna(subset=['AQI']).tail(1)
                        if not latest_row.empty:
                            st.markdown('### Historical Context')
                            h1, h2, h3 = st.columns(3)
                            h1.metric('Historical Mean AQI', f"{historical_avg:.1f}")
                            h2.metric('Historical Peak AQI', f"{history_df['AQI'].max():.1f}")
                            h3.metric('Last Dataset AQI', f"{latest_row['AQI'].iloc[0]:.1f}")
                    else:
                        avg_col.metric('Vs Historical Mean', 'N/A')

                    st.session_state.latest_report = build_prediction_report(
                        selected_city,
                        display_aqi,
                        status_text,
                        'Local Model',
                        pollutant_values,
                    )
                else:
                    display_aqi = float(aqi)
                    st.write(f"Live AQI in {selected_city}: {display_aqi:.2f}")
                    st.info(NO_LOCAL_MODEL_MESSAGE)
                    status_text, color = get_aqi_status(display_aqi)
                    st.markdown(
                        f'<p style="font-size: larger;">The live air quality is <span style="color:{color}; font-size: larger;"><strong>{status_text.upper()}.</strong></span></p>',
                        unsafe_allow_html=True,
                    )
                    render_alert_banner(display_aqi)

                    history_df = load_city_history(selected_city)
                    if history_df is not None and 'AQI' in history_df.columns and history_df['AQI'].notna().any():
                        st.markdown('### Historical Context')
                        h1, h2, h3 = st.columns(3)
                        h1.metric('Historical Mean AQI', f"{history_df['AQI'].mean():.1f}")
                        h2.metric('Historical Peak AQI', f"{history_df['AQI'].max():.1f}")
                        h3.metric('Vs Historical Mean', f"{display_aqi - history_df['AQI'].mean():+.1f}")

                    st.session_state.latest_report = build_prediction_report(
                        selected_city,
                        display_aqi,
                        status_text,
                        'WAQI Live',
                        pollutant_values,
                    )

                st.markdown('### Pollutant Snapshot')
                p1, p2, p3 = st.columns(3)
                p4, p5, p6 = st.columns(3)
                p1.metric('PM2.5', f"{pollutant_values['pm25']:.2f}")
                p2.metric('PM10', f"{pollutant_values['pm10']:.2f}")
                p3.metric('Ozone', f"{pollutant_values['o3']:.2f}")
                p4.metric('Nitrogen Dioxide', f"{pollutant_values['no2']:.2f}")
                p5.metric('Sulfur Dioxide', f"{pollutant_values['so2']:.2f}")
                p6.metric('Carbon Monoxide', f"{pollutant_values['co']:.2f}")

                snapshot_df = build_snapshot_dataframe(
                    selected_city,
                    display_aqi,
                    pollutant_values,
                    'Local Model' if selected_city in city_models else 'WAQI Live',
                )
                st.download_button(
                    'Download AQI Snapshot',
                    data=snapshot_df.to_csv(index=False),
                    file_name='aqi_snapshot.csv',
                    mime='text/csv',
                )
                if st.session_state.latest_report is not None:
                    st.download_button(
                        'Download Prediction Report (JSON)',
                        data=json.dumps(st.session_state.latest_report, indent=2),
                        file_name='prediction_report.json',
                        mime='application/json',
                    )

                render_forecast_panel(selected_city, pollutant='AQI', months_ahead=6)
        else:
            error_message = json_data.get('data')
            if isinstance(error_message, dict):
                error_message = error_message.get('message', 'Unable to fetch AQI data for this city.')
            elif not isinstance(error_message, str):
                error_message = 'Unable to fetch AQI data for this city.'
            st.error(error_message)

elif page == "Health Prediction":
    st.title("Health Prediction")

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
            unsafe_allow_html=True
        )

    add_bg_from_local(local_file('bg3.jpg'))

    st.markdown(
        """
        <div style="background: rgba(8,14,26,0.72); border:1px solid rgba(255,255,255,0.18); border-radius:14px; padding:0.8rem 0.9rem; color:#eef6ff; margin-bottom:0.8rem;">
            <b>Health Model v2 Active:</b> AQI-driven ensemble model with AQI-band engineering.
            Automatically falls back to legacy model files when needed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_city = st.session_state.selected_city if st.session_state.selected_city else "Not selected"
    current_aqi = st.session_state.aqi

    c1, c2 = st.columns(2)
    c1.metric("Selected City", selected_city)
    c2.metric("Current AQI", current_aqi if current_aqi is not None else "N/A")

    if current_aqi is None:
        st.warning("Please run AQI Prediction first to generate current AQI for health forecasting.")
        st.stop()

    def load_model_with_fallback(primary_name, fallback_name):
        primary_path = local_file(primary_name)
        if primary_path.exists():
            try:
                return patch_sklearn_pickle_compat(joblib.load(primary_path)), primary_name
            except Exception:
                # If a custom transformer/function from training is unavailable at runtime,
                # safely fallback to legacy models instead of breaking the page.
                pass
        return patch_sklearn_pickle_compat(joblib.load(local_file(fallback_name))), fallback_name

    model_health, health_model_file = load_model_with_fallback('model_health_v2.pkl', 'model_health.pkl')
    model_general, general_model_file = load_model_with_fallback('model_general_v2.pkl', 'model_general.pkl')
    model_vulnerable, vulnerable_model_file = load_model_with_fallback('model_vulnerable_v2.pkl', 'model_vulnerable.pkl')

    st.caption(
        f"Models in use: {health_model_file}, {general_model_file}, {vulnerable_model_file}"
    )

    input_sample = [[float(current_aqi)]]

    overall_prediction = model_health.predict(input_sample)[0]
    general_prediction = model_general.predict(input_sample)[0]
    vulnerable_prediction = model_vulnerable.predict(input_sample)[0]

    st.subheader("Prediction Summary")
    p1, p2, p3 = st.columns(3)
    p1.info(f"Overall: {overall_prediction}")
    p2.success(f"General Population: {general_prediction}")
    p3.error(f"Vulnerable Population: {vulnerable_prediction}")

    band_text, band_color = get_aqi_status(float(current_aqi))
    st.markdown(
        f'<div style="background: rgba(8,14,26,0.72); border:1px solid rgba(255,255,255,0.18); border-radius:14px; padding:0.8rem 0.9rem; color:#eef6ff; margin:0.8rem 0;">'
        f'<b>Current AQI band:</b> <span style="color:{band_color}; font-weight:700;">{band_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.subheader('Recommended Actions')
    for action in get_health_actions(float(current_aqi)):
        st.write(f"- {action}")

    st.subheader('Advanced Personal Protection Planner')
    planner_col1, planner_col2, planner_col3 = st.columns(3)
    age_group = planner_col1.selectbox('Age Group', ['Child (<18)', 'Adult (18-59)', 'Senior (60+)'])
    daily_outdoor_hours = planner_col2.slider('Daily Outdoor Exposure (hours)', 0.0, 12.0, 2.0, 0.5)
    mask_type = planner_col3.selectbox('Primary Mask Type', ['None', 'Cloth', 'Surgical', 'N95/FFP2'])

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        has_asthma = st.checkbox('Asthma or chronic lung condition')
        has_heart_disease = st.checkbox('Cardiac condition')
    with risk_col2:
        is_pregnant = st.checkbox('Pregnant')
        has_purifier = st.checkbox('Indoor air purifier available')

    sensitivity_points = 0
    if age_group in ['Child (<18)', 'Senior (60+)']:
        sensitivity_points += 2
    if has_asthma:
        sensitivity_points += 3
    if has_heart_disease:
        sensitivity_points += 3
    if is_pregnant:
        sensitivity_points += 2

    personal_risk_score, personal_risk_level = calculate_personal_aqi_risk(
        float(current_aqi),
        daily_outdoor_hours,
        sensitivity_points,
        mask_type,
        has_purifier,
    )
    exposure_load = round(float(current_aqi) * float(daily_outdoor_hours), 1)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric('Personal AQI Risk Score', f"{personal_risk_score}/100")
    metric_col2.metric('Risk Level', personal_risk_level)
    metric_col3.metric('Exposure Load (AQI x hours)', exposure_load)
    st.progress(min(1.0, personal_risk_score / 100.0))

    st.markdown('### Personalized Protection Actions')
    personalized_actions = build_personal_protection_actions(
        float(current_aqi),
        personal_risk_level,
        daily_outdoor_hours,
        mask_type,
        has_purifier,
    )
    for action in personalized_actions:
        st.write(f"- {action}")

    st.markdown('### Safe Outdoor Time Suggestions')
    safe_time_df = build_safe_outdoor_time_plan(float(current_aqi), personal_risk_level)
    safe_time_col1, safe_time_col2 = st.columns(2)
    with safe_time_col1:
        st.dataframe(safe_time_df, use_container_width=True)
    with safe_time_col2:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(safe_time_df['time_window'], safe_time_df['recommended_minutes'], color='#2a9d8f')
        ax.set_xlabel('Recommended Outdoor Minutes')
        ax.set_title('Safer Exposure Windows')
        ax.grid(axis='x', alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)

    st.download_button(
        'Download Safe Time Plan (CSV)',
        data=safe_time_df.to_csv(index=False),
        file_name='safe_outdoor_time_plan.csv',
        mime='text/csv',
    )

    st.markdown('### School or Office Commute Safety Mode')
    commute_col1, commute_col2, commute_col3 = st.columns(3)
    commute_group = commute_col1.selectbox('Commute Profile', ['General', 'School', 'Office'])
    commute_minutes = commute_col2.slider('One-way Commute Time (minutes)', 5, 120, 35, 5)
    travel_mode = commute_col3.selectbox(
        'Primary Travel Mode',
        ['Walking', 'Two-wheeler', 'Public Transport', 'Car (windows closed)', 'School Bus'],
    )

    commute_plan = build_commute_safety_plan(
        float(current_aqi),
        personal_risk_level,
        commute_group,
        commute_minutes,
        travel_mode,
    )

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric('Commute Exposure Load', commute_plan['commute_load'])
    cm2.metric('Tolerance Threshold', commute_plan['tolerated_load'])
    cm3.metric('Commute Status', commute_plan['commute_status'])

    for action in commute_plan['key_actions']:
        st.write(f"- {action}")

    st.markdown('#### Better Commute Time Windows')
    st.dataframe(commute_plan['route_windows'], use_container_width=True)

    st.markdown('### Symptom Check and Urgency')
    selected_symptoms = st.multiselect(
        'Select any symptoms you are noticing right now',
        [
            'Persistent cough',
            'Eye irritation',
            'Throat irritation',
            'Headache',
            'Shortness of breath',
            'Chest pain',
            'Severe wheezing',
            'Faintness',
        ],
    )
    urgency_level, urgency_text = evaluate_symptom_urgency(float(current_aqi), selected_symptoms)
    if urgency_level == 'urgent':
        st.error(urgency_text)
    elif urgency_level == 'high':
        st.warning(urgency_text)
    elif urgency_level == 'medium':
        st.info(urgency_text)
    else:
        st.success(urgency_text)

    if hasattr(model_health, "predict_proba"):
        probs = model_health.predict_proba(input_sample)[0]
        labels = model_health.classes_
        best_idx = int(probs.argmax())
        st.caption(f"Overall model confidence: {labels[best_idx]} ({probs[best_idx] * 100:.1f}%)")

    health_report = build_prediction_report(
        selected_city,
        current_aqi,
        band_text,
        'Health Prediction',
        {},
        overall_prediction=overall_prediction,
        general_prediction=general_prediction,
        vulnerable_prediction=vulnerable_prediction,
    )
    health_report['personal_planner'] = {
        'age_group': age_group,
        'daily_outdoor_hours': daily_outdoor_hours,
        'mask_type': mask_type,
        'has_asthma_or_lung_condition': has_asthma,
        'has_cardiac_condition': has_heart_disease,
        'is_pregnant': is_pregnant,
        'has_air_purifier': has_purifier,
        'personal_risk_score': personal_risk_score,
        'personal_risk_level': personal_risk_level,
        'exposure_load': exposure_load,
        'selected_symptoms': selected_symptoms,
        'symptom_urgency_level': urgency_level,
        'symptom_guidance': urgency_text,
        'personalized_actions': personalized_actions,
        'safe_time_plan': safe_time_df.to_dict(orient='records'),
        'commute_safety': {
            'group_mode': commute_group,
            'one_way_commute_minutes': commute_minutes,
            'travel_mode': travel_mode,
            'commute_load': commute_plan['commute_load'],
            'tolerated_load': commute_plan['tolerated_load'],
            'commute_status': commute_plan['commute_status'],
            'key_actions': commute_plan['key_actions'],
            'route_windows': commute_plan['route_windows'].to_dict(orient='records'),
        },
    }
    st.download_button(
        'Download Health Report (JSON)',
        data=json.dumps(health_report, indent=2),
        file_name='health_report.json',
        mime='application/json',
    )

    report_file = local_file('health_models_report_v2.json')
    if report_file.exists():
        with st.expander("Model Performance (v2)"):
            report = json.loads(report_file.read_text(encoding='utf-8'))
            st.write(f"Training rows: {report.get('dataset_rows', 'N/A')}")
            model_rows = report.get('models', [])
            if model_rows:
                perf_df = pd.DataFrame(model_rows)[['target', 'accuracy', 'f1_weighted', 'model_file']]
                st.dataframe(perf_df, use_container_width=True)


if page == "Analysis":
    st.title("Analysis")
    st.write(f"Selected City: {st.session_state.selected_city}")

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_handle:
            encoded_string = base64.b64encode(image_handle.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image:
                radial-gradient(circle at 12% 20%, rgba(13, 148, 136, 0.28), rgba(13, 148, 136, 0) 36%),
                radial-gradient(circle at 88% 18%, rgba(245, 158, 11, 0.25), rgba(245, 158, 11, 0) 34%),
                linear-gradient(160deg, rgba(10, 20, 38, 0.88), rgba(6, 13, 24, 0.78)),
                url(data:image/{{"jpg"}};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )

    analysis_bg_path = local_file('analysis_bg.jpg')
    if not analysis_bg_path.exists():
        analysis_bg_path = local_file('bg.jpg')
    add_bg_from_local(analysis_bg_path)

    st.markdown(
        """
        <style>
        .vv-analytics {
            background: rgba(7, 13, 24, 0.70);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.75rem;
            color: #edf5ff;
        }
        .vv-analytics b {
            color: #d8ecff;
        }
        </style>
        <div class="vv-analytics">
            <b>Analytics Modes:</b> Year-wise trend, Month-wise comparison, and Monthly continuous trend.
            Forecast extension is enabled up to 2026 for better directional insights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.selected_city not in city_data_map:
        st.warning("Please select a city in AQI Prediction first.")
        st.stop()

    city_file = city_data_map[st.session_state.selected_city]
    city_file_path = local_file(city_file)

    if not city_file_path.exists():
        st.warning(
            f"Historical dataset not found for {st.session_state.selected_city}. "
            "Add the city CSV file to enable analysis graphs."
        )
        st.stop()



    analysis_df = load_city_history(st.session_state.selected_city)
    render_all_cities_top_aqi_months(start_year=2013, end_year=2026, top_n=12)
    render_analysis_overview(analysis_df)

    # Radio button for selecting pollutants
    selected_pollutant = st.radio("Select Pollutant", ["pm25", "pm10", "o3", "no2", "so2", "co", "AQI"])

    # Radio buttons for selecting analysis type
    selected_analysis = st.radio(
        "Select Analysis Type",
        ["Year-wise Analysis", "Month-wise Analysis", "Monthly Trend Graph", "City Comparison", "Forecast Outlook"]
    )

    if selected_analysis == "Year-wise Analysis":
        chart_obj = process_pollutant(analysis_df.copy(), selected_pollutant, selected_pollutant.upper(), extend_to_year=2026)
        st.caption("Graph is extended to 2026 using trend-based projection for missing future years.")
        st.pyplot(chart_obj)

    # Add similar conditions for Month-wise and Day-wise Analysis
    elif selected_analysis == "Month-wise Analysis":
        chart_obj = monthly_analysis(analysis_df.copy(), selected_pollutant, extend_to_year=2026)
        st.caption("Graph is extended to 2026 using trend-based projection for missing future months.")
        st.pyplot(chart_obj)

    elif selected_analysis == "Monthly Trend Graph":
        chart_obj = monthly_trend_analysis(analysis_df.copy(), selected_pollutant, extend_to_year=2026)
        st.caption("Continuous month-wise trend graph extended through 2026.")
        st.pyplot(chart_obj)

    elif selected_analysis == "City Comparison":
        st.caption("Compare the selected pollutant across all cities with available historical datasets.")
        render_city_comparison(selected_pollutant)

    elif selected_analysis == "Forecast Outlook":
        st.caption("Forecast uses month-wise historical trend projection from the local city dataset.")
        render_forecast_panel(st.session_state.selected_city, pollutant=selected_pollutant, months_ahead=6)

