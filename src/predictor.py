from datetime import timedelta

import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, hourly_model, daily_model, hourly_scaler, daily_scaler, data_processor):
        self.hourly_model = hourly_model
        self.daily_model = daily_model
        self.hourly_scaler = hourly_scaler
        self.daily_scaler = daily_scaler
        self.data_processor = data_processor
        self.hourly_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'year']
        self.daily_features = ['day_of_week', 'day_of_month', 'month', 'year']

    def predict(self, last_timestamp, mode='both'):
        """Generate predictions"""
        predictions = {}

        if mode in ['hourly', 'both']:
            # Simplify to single prediction per day
            hourly_data = self.data_processor.prepare_prediction_data(last_timestamp, 'hourly')
            X_hourly = hourly_data[self.hourly_features]
            X_hourly_scaled = self.hourly_scaler.transform(X_hourly)

            # Predict with multiple sample points, but limit to 24
            hourly_predictions = []
            for hour in range(24):
                # Create a slightly varied input for each hour
                X_hour = X_hourly.copy()
                X_hour['hour'] = hour
                X_hour_scaled = self.hourly_scaler.transform(X_hour)

                # Predict with small randomness
                pred = self.hourly_model.predict(X_hour_scaled)
                randomness = np.random.normal(0, 0.5)
                hourly_predictions.append(pred[0] + randomness)

            # Create DataFrame with hourly predictions
            hourly_data['predicted_value'] = hourly_predictions
            hourly_data = hourly_data.iloc[0:24].copy()  # Limit to 24 rows
            hourly_data['timestamp'] = [last_timestamp.replace(hour=h, minute=0, second=0, microsecond=0) for h in
                                        range(24)]

            predictions['hourly'] = hourly_data

        if mode in ['daily', 'both']:
            daily_data = self.data_processor.prepare_prediction_data(last_timestamp, 'daily')
            X_daily = daily_data[self.daily_features]
            X_daily_scaled = self.daily_scaler.transform(X_daily)
            daily_predictions = self.daily_model.predict(X_daily_scaled)
            daily_data['predicted_value'] = daily_predictions
            predictions['daily'] = daily_data

        return predictions