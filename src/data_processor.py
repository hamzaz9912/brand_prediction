import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataProcessor:
    @staticmethod
    def process_data(df):
        """Process the uploaded data for model training"""
        try:
            # Identify datetime column
            date_column_candidates = ['timestamp', 'date', 'datetime', 'time', 'Date', 'DateTime', 'Time']
            datetime_col = None

            for col in date_column_candidates:
                if col in df.columns:
                    datetime_col = col
                    break

            if datetime_col is None:
                datetime_col = df.columns[0]

            # Identify value column
            value_column_candidates = ['value', 'price', 'amount', 'Value', 'Price', 'Amount', 'Close']
            value_col = None

            for col in value_column_candidates:
                if col in df.columns:
                    value_col = col
                    break

            if value_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                value_col = numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[1]

            # Create standardized DataFrame
            standardized_df = pd.DataFrame()
            standardized_df['timestamp'] = pd.to_datetime(df[datetime_col])
            standardized_df['value'] = pd.to_numeric(df[value_col], errors='coerce')

            # Create features for daily and hourly predictions
            standardized_df['hour'] = standardized_df['timestamp'].dt.hour
            standardized_df['day_of_week'] = standardized_df['timestamp'].dt.dayofweek
            standardized_df['day_of_month'] = standardized_df['timestamp'].dt.day
            standardized_df['month'] = standardized_df['timestamp'].dt.month
            standardized_df['year'] = standardized_df['timestamp'].dt.year

            # Create daily average values
            daily_df = standardized_df.resample('D', on='timestamp')['value'].agg(['mean', 'min', 'max']).reset_index()
            daily_df.columns = ['timestamp', 'daily_mean', 'daily_min', 'daily_max']

            # Merge daily values back
            standardized_df = pd.merge(
                standardized_df,
                daily_df,
                on='timestamp',
                how='left'
            )

            return standardized_df.dropna().sort_values('timestamp')

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