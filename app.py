# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from config import CONFIG
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from utils.helpers import save_uploaded_file, load_brand_data
import pytz

class StockDataProcessor:
    def __init__(self):
        self.trading_hours = {
            'start': '09:00',  # Market opens at 9:00 AM
            'end': '15:00'  # Market closes at 3:00 PM
        }

    def generate_hourly_prices(self, open_price, close_price, high, low, timestamp):
        """Generate synthetic hourly prices with increased randomness and market dynamics"""
        num_points = 6  # Only 6 points from 9 AM to 3 PM
        np.random.seed(int(timestamp.timestamp()))  # Seed for reproducibility

        # Create price trajectory with more dynamic variation
        hours = np.linspace(9, 15, num=num_points)  # From 9 AM to 3 PM
        prices = np.zeros(num_points)

        # Start and end points
        prices[0] = open_price
        prices[-1] = close_price

        # Price range and volatility
        price_range = high - low
        volatility = price_range * 0.15  # Increased volatility to 15%

        # Market session dynamics
        def market_hour_effect(hour):
            """Simulate typical market hour price movements"""
            # More volatility in market opening and closing hours
            market_hours = [(0, 2), (9, 11), (21, 22)]  # Early morning, mid-morning, late afternoon
            for start, end in market_hours:
                if start <= hour <= end:
                    return 1.5  # Higher volatility during these periods
            return 1.0

        # Generate intermediate prices
        for i in range(1, num_points - 1):
            # Progress through trading day
            progress = (hours[i] - 9) / 6  # Scale to progress from 9 to 15 hours

            # Base interpolation between open and close with non-linear progression
            base_price = open_price * (1 - np.sin(progress * np.pi)) + close_price * np.sin(progress * np.pi)

            # Market hour effect
            hour_effect = market_hour_effect(hours[i])

            # Add randomness with different distribution
            price_noise = np.random.normal(0, volatility * hour_effect) * np.sin(progress * np.pi)
            momentum = np.random.uniform(-volatility, volatility) * hour_effect
            trend_component = (close_price - open_price) * progress * 0.5

            # Calculate price with multiple randomness factors
            new_price = base_price + price_noise + momentum + trend_component

            # Constrain within day's range with some flex
            new_price = max(min(new_price, high * 1.02), low * 0.98)
            prices[i] = new_price

        # Generate timestamps for the selected time range
        tz = pytz.timezone("Asia/Karachi")
        base_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0).astimezone(tz)

        timestamps = [base_time + timedelta(hours=h - 9) for h in hours]  # Adjust for 9 AM as start time

        return timestamps, prices

    def process_stock_data(self, df):
        """Convert daily OHLC data to hourly data with interpolation and increased realism"""
        # Ensure datetime conversion
        df['timestamp'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC').dt.tz_convert('Asia/Karachi')
        df = df.sort_values('timestamp')

        # Container for hourly data
        hourly_data = []

        # Process each unique date separately
        for date in df['Date'].unique():
            # Get row for specific date
            row = df[df['Date'] == date].iloc[0]

            # Get prices and timestamp
            open_price = float(row['Open'])
            close_price = float(row['Price'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            timestamp = row['timestamp']

            # Generate hourly timestamps and prices
            hourly_timestamps, hourly_prices = self.generate_hourly_prices(
                open_price, close_price, high_price, low_price, timestamp
            )

            # Create hourly records
            for t, v in zip(hourly_timestamps, hourly_prices):
                hourly_data.append({
                    'timestamp': t,
                    'value': v,
                    'original_date': date,
                    'is_market_hour': True
                })

        # Create DataFrame with generated hourly data
        hourly_df = pd.DataFrame(hourly_data)
        hourly_df = hourly_df.sort_values('timestamp')

        # Add technical indicators
        hourly_df = self.add_technical_indicators(hourly_df)

        return hourly_df
    def add_technical_indicators(self, df):
        """Add technical indicators to the hourly data"""
        # Simple Moving Averages
        df['SMA_5'] = df['value'].rolling(window=5).mean()
        df['SMA_20'] = df['value'].rolling(window=20).mean()

        # Relative Strength Index (RSI)
        delta = df['value'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['value'].rolling(window=20).mean()
        bb_std = df['value'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        # MACD
        exp1 = df['value'].ewm(span=12, adjust=False).mean()
        exp2 = df['value'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

# Example usage:
def prepare_stock_data(csv_file):
    """Prepare stock data from CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize processor
    processor = StockDataProcessor()

    # Process data
    hourly_df = processor.process_stock_data(df)

    return hourly_df
def create_historical_graph(df, predictions, brand, start_date, end_date):
    """Create historical graph with future predictions for selected date range"""
    fig = go.Figure()

    # Filter data for selected date range
    mask = ((df['timestamp'].dt.date >= start_date) &
            (df['timestamp'].dt.date <= end_date))
    filtered_df = df[mask]

    # Filter predictions
    pred_mask = ((predictions['daily']['timestamp'].dt.date >= start_date) &
                 (predictions['daily']['timestamp'].dt.date <= end_date))
    filtered_predictions = predictions['daily'][pred_mask]

    # Historical data
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['value'],
        name='Historical Data',
        line=dict(color='blue', width=2, shape='spline'),
        mode='lines'
    ))

    # Future predictions
    fig.add_trace(go.Scatter(
        x=filtered_predictions['timestamp'],
        y=filtered_predictions['predicted_value'],
        name='Future Predictions',
        line=dict(color='red', width=2, dash='dash', shape='spline'),
        mode='lines'
    ))

    fig.update_layout(
        title=dict(
            text=f"{brand} - Historical Data and Predictions ({start_date} to {end_date})",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey',
            showgrid=True
        ),
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white'
    )

    return fig


def create_daily_prediction_graph(df, predictions, selected_date, brand):
    """Create daily prediction graph with smooth lines"""
    fig = go.Figure()

    # Filter data for selected date
    daily_data = df[df['timestamp'].dt.date == selected_date]
    daily_predictions = predictions['daily'][predictions['daily']['timestamp'].dt.date == selected_date]

    # Plot actual data with smooth line
    if not daily_data.empty:
        fig.add_trace(go.Scatter(
            x=daily_data['timestamp'],
            y=daily_data['value'],
            name='Actual Data',
            line=dict(color='blue', width=2, shape='spline'),
            mode='lines'
        ))

    # Plot predictions with smooth line
    if not daily_predictions.empty:
        fig.add_trace(go.Scatter(
            x=daily_predictions['timestamp'],
            y=daily_predictions['predicted_value'],
            name='Predictions',
            line=dict(color='red', width=2, dash='dash', shape='spline'),
            mode='lines'
        ))

    fig.update_layout(
        title=dict(
            text=f"{brand} - Daily View ({selected_date})",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Time",
            tickformat='%H:%M',
            dtick=3600000,  # 1 hour in milliseconds
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey',
            showgrid=True
        ),
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white'
    )

    return fig

def create_hourly_prediction_graph(df, predictions, selected_date, brand):
    """Create hourly prediction graph with smooth lines (only showing 6 points from 9 AM to 3 PM)"""
    fig = go.Figure()

    # Ensure consistent timezone
    tz = pytz.timezone('Asia/Karachi')
    selected_date = pd.Timestamp(selected_date).tz_localize(tz)

    # Convert timestamp columns to timezone-aware datetimes
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.tz_localize('Asia/Karachi')
    predictions['hourly']['timestamp'] = pd.to_datetime(predictions['hourly']['timestamp']).dt.tz_localize(None).dt.tz_localize('Asia/Karachi')

    # Filter data for selected date using date comparison
    hourly_data = df[df['timestamp'].dt.date == selected_date.date()]
    hourly_predictions = predictions['hourly'][predictions['hourly']['timestamp'].dt.date == selected_date.date()]

    # Filter the hourly data to only include times between 9 AM and 3 PM
    hourly_data = hourly_data[(hourly_data['timestamp'].dt.hour >= 9) & (hourly_data['timestamp'].dt.hour <= 15)]
    hourly_predictions = hourly_predictions[(hourly_predictions['timestamp'].dt.hour >= 9) & (hourly_predictions['timestamp'].dt.hour <= 15)]

    # Ensure we sample exactly 6 points
    if len(hourly_data) >= 6:
        hourly_data = hourly_data.iloc[np.linspace(0, len(hourly_data)-1, 6, dtype=int)]  # Sample 6 points
        hourly_predictions = hourly_predictions.iloc[np.linspace(0, len(hourly_predictions)-1, 6, dtype=int)]  # Sample 6 points
    else:
        # If there are fewer than 6 points, use all available data (adjust logic as needed)
        hourly_data = hourly_data.iloc[np.linspace(0, len(hourly_data)-1, len(hourly_data), dtype=int)]
        hourly_predictions = hourly_predictions.iloc[np.linspace(0, len(hourly_predictions)-1, len(hourly_predictions), dtype=int)]

    if not hourly_data.empty:
        fig.add_trace(go.Scatter(
            x=hourly_data['timestamp'],
            y=hourly_data['value'],
            name='Actual Data',
            line=dict(color='blue', width=2, shape='spline'),
            mode='lines'
        ))

    if not hourly_predictions.empty:
        fig.add_trace(go.Scatter(
            x=hourly_predictions['timestamp'],
            y=hourly_predictions['predicted_value'],
            name='Predictions',
            line=dict(color='red', width=2, dash='dash', shape='spline'),
            mode='lines'
        ))

    fig.update_layout(
        title=dict(
            text=f"{brand} - Hourly Predictions ({selected_date.date()})",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Time",
            tickformat='%H:%M',
            dtick=3600000,  # 1 hour in milliseconds
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey',
            showgrid=True
        ),
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white'
    )

    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Brand Prediction Dashboard")

    # Initialize classes
    data_processor = DataProcessor()
    model_trainer = ModelTrainer(CONFIG['MODELS_DIR'])

    # Sidebar layout
    st.sidebar.header("Settings")
    selected_brand = st.sidebar.selectbox("Select Brand", CONFIG['AVAILABLE_BRANDS'])

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        f"Upload new data for {selected_brand}",
        type=['csv']
    )

    try:
        if uploaded_file:
            st.info("Processing uploaded file...")
            df = save_uploaded_file(uploaded_file, selected_brand, CONFIG['DATA_DIR'])

            if df is not None:
                processed_df = data_processor.process_data(df)
                hourly_model, hourly_scaler = model_trainer.train_model(processed_df, selected_brand, 'hourly')
                daily_model, daily_scaler = model_trainer.train_model(processed_df, selected_brand, 'daily')
                st.success(f"Model trained successfully for {selected_brand}")

        # Load data and models
        df = load_brand_data(selected_brand, CONFIG['DATA_DIR'])
        hourly_model, hourly_scaler = model_trainer.load_model(selected_brand, 'hourly')
        daily_model, daily_scaler = model_trainer.load_model(selected_brand, 'daily')

        if df is not None and hourly_model is not None and daily_model is not None:
            processed_df = data_processor.process_data(df)

            # Date range selection in sidebar
            st.sidebar.header("Date Range Selection")
            min_date = processed_df['timestamp'].dt.date.min()
            max_date = processed_df['timestamp'].dt.date.max()

            # Date range selection with two columns
            date_cols = st.sidebar.columns(2)
            with date_cols[0]:
                start_date = st.date_input(
                    "Start Date",
                    value=max_date - timedelta(days=7),
                    min_value=min_date,
                    max_value=max_date
                )

            with date_cols[1]:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=start_date,
                    max_value=max_date + timedelta(days=30)
                )

            if start_date > end_date:
                st.error("Error: End date must be after start date")
                return

            # Selected date for daily/hourly view
            selected_date = st.sidebar.date_input(
                "Select Date for Daily/Hourly View",
                value=end_date,
                min_value=start_date,
                max_value=end_date
            )

            # Initialize predictor
            predictor = Predictor(
                hourly_model, daily_model,
                hourly_scaler, daily_scaler,
                data_processor
            )

            last_timestamp = processed_df['timestamp'].max()
            predictions = predictor.predict(last_timestamp, 'both')

            # Filter data for selected date range
            mask = ((processed_df['timestamp'].dt.date >= start_date) &
                    (processed_df['timestamp'].dt.date <= end_date))
            date_range_df = processed_df[mask]

            # Display metrics for selected range
            st.subheader(f"Metrics ({start_date} to {end_date})")
            metric_cols = st.columns(4)

            with metric_cols[0]:
                range_avg = date_range_df['value'].mean()
                st.metric("Average Value", f"{range_avg:.2f}")

            with metric_cols[1]:
                range_max = date_range_df['value'].max()
                st.metric("Maximum Value", f"{range_max:.2f}")

            with metric_cols[2]:
                range_min = date_range_df['value'].min()
                st.metric("Minimum Value", f"{range_min:.2f}")

            with metric_cols[3]:
                value_change = date_range_df['value'].iloc[-1] - date_range_df['value'].iloc[0]
                st.metric("Value Change", f"{value_change:.2f}")

            # Display graphs
            st.subheader("Historical View")
            historical_fig = create_historical_graph(processed_df, predictions, selected_brand,
                                                     start_date, end_date)
            st.plotly_chart(historical_fig, use_container_width=True)

            # Daily and Hourly Predictions
            graph_cols = st.columns(2)

            with graph_cols[0]:
                daily_fig = create_daily_prediction_graph(processed_df, predictions,
                                                          selected_date, selected_brand)
                st.plotly_chart(daily_fig, use_container_width=True)

            with graph_cols[1]:
                hourly_fig = create_hourly_prediction_graph(processed_df, predictions,
                                                            selected_date, selected_brand)
                st.plotly_chart(hourly_fig, use_container_width=True)

        else:
            st.warning(f"No data available for {selected_brand}. Please upload data.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure your CSV file contains at least two columns: a date/time column and a value column.")


if __name__ == "__main__":
    main()