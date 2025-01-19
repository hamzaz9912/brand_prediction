import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataProcessor:
    @staticmethod
    def process_data(df):
        """Process the uploaded data for model training"""
        try:
            # Check for the date column
            if 'mDate' in df.columns:
                df = df.rename(columns={'mDate': 'timestamp'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            else:
                raise Exception("No date column found. Ensure the date column is named `mDate` or `Date`.")

            # Parse the `timestamp` column as datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y', errors='coerce')

            # Drop rows with invalid or missing dates
            df = df.dropna(subset=['timestamp'])
            if df.empty:
                raise Exception(
                    "No valid dates found in the date column. Check the date format (expected: MM/DD/YYYY).")

            # Clean and process the `Price` column
            if 'Price' in df.columns:
                # Remove commas and convert to numeric
                df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
            else:
                raise Exception("No `Price` column found in the file. Ensure the value column is named `Price`.")

            # Drop rows with missing or invalid prices
            df = df.dropna(subset=['Price'])
            if df.empty:
                raise Exception("No valid prices found in the `Price` column.")

            # Standardize the DataFrame
            standardized_df = pd.DataFrame()
            standardized_df['timestamp'] = df['timestamp']
            standardized_df['value'] = df['Price']

            # Create features for daily and hourly predictions
            standardized_df['hour'] = standardized_df['timestamp'].dt.hour
            standardized_df['day_of_week'] = standardized_df['timestamp'].dt.dayofweek
            standardized_df['day_of_month'] = standardized_df['timestamp'].dt.day
            standardized_df['month'] = standardized_df['timestamp'].dt.month
            standardized_df['year'] = standardized_df['timestamp'].dt.year

            # Create daily statistics
            daily_df = standardized_df.resample('D', on='timestamp')['value'].agg(['mean', 'min', 'max']).reset_index()
            daily_df.columns = ['timestamp', 'daily_mean', 'daily_min', 'daily_max']

            # Merge daily statistics back to the main DataFrame
            standardized_df = pd.merge(
                standardized_df,
                daily_df,
                on='timestamp',
                how='left'
            )

            return standardized_df.sort_values('timestamp')

        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

    @staticmethod
    def prepare_prediction_data(last_timestamp, mode='hourly'):
        """Prepare data for predictions"""
        future_dates = []
        current = last_timestamp

        if mode == 'hourly':
            intervals = 24  # 24 hours
            delta = timedelta(hours=1)
        else:  # daily
            intervals = 30  # 30 days
            delta = timedelta(days=1)

        for _ in range(intervals):
            current += delta
            future_dates.append({
                'timestamp': current,
                'hour': current.hour,
                'day_of_week': current.dayofweek,
                'day_of_month': current.day,
                'month': current.month,
                'year': current.year
            })

        return pd.DataFrame(future_dates)


