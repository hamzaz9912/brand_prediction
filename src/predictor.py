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
            hourly_data = self.data_processor.prepare_prediction_data(last_timestamp, 'hourly')
            X_hourly = hourly_data[self.hourly_features]
            X_hourly_scaled = self.hourly_scaler.transform(X_hourly)
            hourly_predictions = self.hourly_model.predict(X_hourly_scaled)
            hourly_data['predicted_value'] = hourly_predictions
            predictions['hourly'] = hourly_data

        if mode in ['daily', 'both']:
            daily_data = self.data_processor.prepare_prediction_data(last_timestamp, 'daily')
            X_daily = daily_data[self.daily_features]
            X_daily_scaled = self.daily_scaler.transform(X_daily)
            daily_predictions = self.daily_model.predict(X_daily_scaled)
            daily_data['predicted_value'] = daily_predictions
            predictions['daily'] = daily_data

        return predictions