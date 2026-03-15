import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def process_pollutant(df, pollutant_column, plot_title, extend_to_year=None):
    # Convert 'NaN' values to the mean of the column
    df[pollutant_column] = pd.to_numeric(df[pollutant_column], errors='coerce')
    df[pollutant_column] = df[pollutant_column].fillna(df[pollutant_column].mean())

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract year and pollutant levels
    df['year'] = df['date'].dt.year

    # Calculate the mean pollutant levels for each year
    yearly_means = df.groupby('year')[pollutant_column].mean().sort_index()

    forecast_years = []
    if extend_to_year is not None and not yearly_means.empty:
        last_year = int(yearly_means.index.max())
        if extend_to_year > last_year:
            x = yearly_means.index.to_numpy(dtype=float)
            y = yearly_means.to_numpy(dtype=float)
            if len(x) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                for year in range(last_year + 1, int(extend_to_year) + 1):
                    yearly_means.loc[year] = float(intercept + slope * year)
                    forecast_years.append(year)
            else:
                base_val = float(y[0])
                for year in range(last_year + 1, int(extend_to_year) + 1):
                    yearly_means.loc[year] = base_val
                    forecast_years.append(year)

    yearly_means = yearly_means.sort_index()

    # Plotting the bar chart
    plt.figure(figsize=(12, 6))
    colors = []
    for year in yearly_means.index:
        if int(year) in forecast_years:
            colors.append('#f4a261')
        else:
            colors.append('#2a9d8f')
    yearly_means.plot(kind='bar', color=colors)

    plt.title(f'Year-wise Mean {plot_title} Levels')
    plt.xlabel('Year')
    plt.ylabel(f'Mean {plot_title} Levels')
    if forecast_years:
        plt.legend(['Actual + Forecast (forecast highlighted in orange)'])

    return plt


def monthly_analysis(df, pollutant_column, extend_to_year=None):
    df = df.copy()

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convert the pollutant column to numeric
    df[pollutant_column] = pd.to_numeric(df[pollutant_column], errors='coerce')

    # Check for non-numeric values
    non_numeric_values = df[pollutant_column][df[pollutant_column].isnull()]
    if not non_numeric_values.empty:
        print(f"Non-numeric values found in {pollutant_column}: {non_numeric_values}")

    # Calculate the mean pollutant levels for each month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    monthly_means = df.groupby(['year', 'month'])[pollutant_column].mean()

    if extend_to_year is not None and not monthly_means.empty:
        monthly_df = monthly_means.reset_index(name='value')
        monthly_df.columns = ['year', 'month', 'value']

        max_year = int(monthly_df['year'].max())
        if extend_to_year > max_year:
            rows = []
            for year in range(max_year + 1, int(extend_to_year) + 1):
                for month in range(1, 13):
                    month_hist = monthly_df[monthly_df['month'] == month]
                    if len(month_hist) >= 2:
                        x = month_hist['year'].to_numpy(dtype=float)
                        y_vals = month_hist['value'].to_numpy(dtype=float)
                        slope, intercept = np.polyfit(x, y_vals, 1)
                        pred = float(intercept + slope * year)
                    elif len(month_hist) == 1:
                        pred = float(month_hist['value'].iloc[0])
                    else:
                        pred = float(monthly_df['value'].mean())
                    rows.append({'year': year, 'month': month, 'value': pred})

            if rows:
                forecast_df = pd.DataFrame(rows)
                monthly_df = pd.concat([monthly_df, forecast_df], ignore_index=True)

            monthly_means = monthly_df.groupby(['year', 'month'])['value'].mean()

    # Plotting the bar chart for monthly analysis
    plt.figure(figsize=(12, 6))
    monthly_means.unstack().plot(kind='bar', title=f'Monthly {pollutant_column.upper()}')
    plt.xlabel('Year-Month')
    plt.ylabel(pollutant_column.upper())
    plt.tight_layout()
    return plt


def monthly_trend_analysis(df, pollutant_column, extend_to_year=None):
    # Build a continuous month-wise timeline for clearer trend visualization.
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df[pollutant_column] = pd.to_numeric(df[pollutant_column], errors='coerce')
    df = df.dropna(subset=['date', pollutant_column])

    monthly_series = (
        df.groupby(pd.Grouper(key='date', freq='MS'))[pollutant_column]
        .mean()
        .sort_index()
    )

    if extend_to_year is not None and not monthly_series.empty:
        last_date = monthly_series.index.max()
        target_date = pd.Timestamp(year=int(extend_to_year), month=12, day=1)
        if target_date > last_date:
            hist_df = monthly_series.reset_index(name='value')
            hist_df['year'] = hist_df['date'].dt.year
            hist_df['month'] = hist_df['date'].dt.month

            rows = []
            current = pd.Timestamp(year=last_date.year, month=last_date.month, day=1) + pd.offsets.MonthBegin(1)
            while current <= target_date:
                month_hist = hist_df[hist_df['month'] == current.month]
                if len(month_hist) >= 2:
                    x = month_hist['year'].to_numpy(dtype=float)
                    y_vals = month_hist['value'].to_numpy(dtype=float)
                    slope, intercept = np.polyfit(x, y_vals, 1)
                    pred = float(intercept + slope * current.year)
                elif len(month_hist) == 1:
                    pred = float(month_hist['value'].iloc[0])
                else:
                    pred = float(hist_df['value'].mean())

                rows.append({'date': current, 'value': pred})
                current = current + pd.offsets.MonthBegin(1)

            if rows:
                forecast_series = pd.Series(
                    {r['date']: r['value'] for r in rows},
                    name=pollutant_column,
                )
                monthly_series = pd.concat([monthly_series, forecast_series]).sort_index()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(monthly_series.index, monthly_series.values, linewidth=2)
    ax.set_title(f'Month-wise Trend {pollutant_column.upper()}')
    ax.set_xlabel('Date')
    ax.set_ylabel(pollutant_column.upper())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return plt
