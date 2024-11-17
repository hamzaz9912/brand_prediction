from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os


class ModelTrainer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.hourly_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'year']
        self.daily_features = ['day_of_week', 'day_of_month', 'month', 'year']

    def train_model(self, df, brand, model_type='hourly'):
        """Train model for specific brand and type"""
        try:
            features = self.hourly_features if model_type == 'hourly' else self.daily_features

            X = df[features]
            if model_type == 'hourly':
                y = df['value']
            else:
                y = df['daily_mean']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_scaled, y)

            # Save model and scaler
            model_path = os.path.join(self.model_dir, f'{brand.lower()}_{model_type}_model.joblib')
            scaler_path = os.path.join(self.model_dir, f'{brand.lower()}_{model_type}_scaler.joblib')

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)

            return model, scaler

        except Exception as e:
            raise Exception(f"Error training {model_type} model: {str(e)}")

    def load_model(self, brand, model_type='hourly'):
        """Load trained model and scaler"""
        try:
            model_path = os.path.join(self.model_dir, f'{brand.lower()}_{model_type}_model.joblib')
            scaler_path = os.path.join(self.model_dir, f'{brand.lower()}_{model_type}_scaler.joblib')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler
            return None, None
        except Exception as e:
            raise Exception(f"Error loading {model_type} model: {str(e)}")