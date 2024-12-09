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

    def predict(self, last_timestamp, mode='both', day_offset=0):
        """
        Generate predictions for current, previous, or next day.

        Parameters:
        - last_timestamp: datetime object, the reference timestamp for prediction.
        - mode: 'hourly', 'daily', or 'both' to select prediction type.
        - day_offset: Integer, 0 for current day, -1 for previous, 1 for next.

        Returns:
        - Dictionary containing hourly and/or daily predictions.
        """
        predictions = {}
        target_date = last_timestamp + timedelta(days=day_offset)

        # Hourly predictions
        if mode in ['hourly', 'both']:
            hourly_data = self.data_processor.prepare_prediction_data(target_date, 'hourly')
            X_hourly = hourly_data[self.hourly_features]

            hourly_predictions = []
            for hour in range(24):
                X_hour = X_hourly.copy()
                X_hour['hour'] = hour
                X_hour_scaled = self.hourly_scaler.transform(X_hour)

                pred = self.hourly_model.predict(X_hour_scaled)
                randomness = np.random.normal(0, 0.5)
                hourly_predictions.append(pred[0] + randomness)

            hourly_data['predicted_value'] = hourly_predictions
            hourly_data['timestamp'] = [target_date.replace(hour=h, minute=0, second=0, microsecond=0) for h in
                                        range(24)]
            predictions['hourly'] = hourly_data

        # Daily predictions
        if mode in ['daily', 'both']:
            daily_data = self.data_processor.prepare_prediction_data(target_date, 'daily')
            X_daily = daily_data[self.daily_features]
            X_daily_scaled = self.daily_scaler.transform(X_daily)

            daily_predictions = self.daily_model.predict(X_daily_scaled)
            daily_data['predicted_value'] = daily_predictions
            predictions['daily'] = daily_data

        return predictions

    def predict_current_day(self, last_timestamp):
        """Predict for the current day (both hourly and daily)."""
        return self.predict(last_timestamp, mode='both', day_offset=0)

    def predict_next_day(self, last_timestamp):
        """Predict for the next day (both hourly and daily)."""
        return self.predict(last_timestamp, mode='both', day_offset=1)

    def predict_previous_day(self, last_timestamp):
        """Predict for the previous day (both hourly and daily)."""
        return self.predict(last_timestamp, mode='both', day_offset=-1)
