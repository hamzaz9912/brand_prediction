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


class StockDataProcessor:
    def __init__(self):
        self.trading_hours = {
            'start': '09:00',  # Market opens at 9:00 AM
            'end': '18:00'      # Market closes at 3:00 PM
        }

    def generate_hourly_prices(self, open_price, close_price, high, low, timestamp):
        """Generate synthetic hourly prices between market open and close"""
        trading_hours = 9  # 6 hours from 9:00 AM to 3:00 PM
        hours = np.linspace(0, trading_hours, num=7)  # 7 points for 6 intervals

        # Create base price trajectory
        prices = []
        for hour in hours:
            # Calculate progress through the trading day (0 to 1)
            progress = hour / trading_hours

            # Base price using weighted average of open and close
            base_price = open_price * (1 - progress) + close_price * progress

            # Add some randomness within the day's range
            price_range = high - low
            random_factor = np.random.normal(0, 0.15)  # Random variation
            price = base_price + random_factor * price_range * 0.1  # 10% of day's range

            # Ensure price stays within day's range
            price = min(max(price, low), high)
            prices.append(price)

        # Ensure first and last prices match open and close
        prices[0] = open_price
        prices[-1] = close_price

        # Generate timestamps
        timestamps = []
        base_time = timestamp.replace(hour=9, minute=0)
        for i in range(len(hours)):
            hour_delta = timedelta(hours=hours[i])
            timestamps.append(base_time + hour_delta)

        return timestamps, prices

    def process_stock_data(self, df):
        """Convert daily OHLC data to hourly data"""
        # Convert date column to datetime if it's not already
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Create hourly data points
        hourly_data = []

        for idx, row in df.iterrows():
            timestamps, prices = self.generate_hourly_prices(
                float(row['Open']),
                float(row['Price']),  # Assuming 'Price' is the closing price
                float(row['High']),
                float(row['Low']),
                row['timestamp']
            )

            # Create hourly records
            for t, p in zip(timestamps, prices):
                hourly_data.append({
                    'timestamp': t,
                    'value': p,
                    'original_date': row['Date'],
                    'is_market_hour': True
                })

        # Create DataFrame from hourly data
        hourly_df = pd.DataFrame(hourly_data)

        # Sort by timestamp
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
    """Create hourly prediction graph with smooth lines"""
    fig = go.Figure()

    # Filter data
    hourly_data = df[df['timestamp'].dt.date == selected_date]
    hourly_predictions = predictions['hourly'][predictions['hourly']['timestamp'].dt.date == selected_date]

    # Plot actual hourly data with smooth line
    if not hourly_data.empty:
        fig.add_trace(go.Scatter(
            x=hourly_data['timestamp'],
            y=hourly_data['value'],
            name='Actual Data',
            line=dict(color='blue', width=2, shape='spline'),
            mode='lines'
        ))

    # Plot hourly predictions with smooth line
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
            text=f"{brand} - Hourly Predictions ({selected_date})",
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


def create_interpolated_hourly_chart(df, brand, selected_date):
    # Filter data for selected date during market hours
    market_start = datetime.combine(selected_date, datetime.strptime('09:00', '%H:%M').time())
    market_end = datetime.combine(selected_date, datetime.strptime('18:00', '%H:%M').time())

    # Create mask for selected date and market hours
    mask = (
            (df['timestamp'].dt.date == selected_date) &
            (df['timestamp'] >= market_start) &
            (df['timestamp'] <= market_end)
    )
    hourly_data = df[mask].copy()

    # If no data, return empty figure
    if hourly_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No Data Available for {brand} on {selected_date}",
            height=400
        )
        return fig

    # Create Figure
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=hourly_data['timestamp'],
        y=hourly_data['value'],
        mode='lines+markers',
        name='Actual Hourly Data',
        line=dict(color='green', width=2),
        marker=dict(size=8, color='green')
    ))

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f"{brand} - Interpolated Hourly Data ({selected_date})",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Time",
            tickformat='%H:%M',
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
        plot_bgcolor='white'
    )

    return fig
# app.py (continued)

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

            # Display historical graph
            st.subheader("Historical View")
            historical_fig = create_historical_graph(processed_df, predictions, selected_brand,
                                                    start_date, end_date)
            st.plotly_chart(historical_fig, use_container_width=True)

            # Daily, Hourly Predictions and Interpolated Hourly Chart
            graph_cols = st.columns(3)  # Changed to 3 columns to accommodate the new chart

            with graph_cols[0]:
                daily_fig = create_daily_prediction_graph(processed_df, predictions,
                                                          selected_date, selected_brand)
                st.plotly_chart(daily_fig, use_container_width=True)

            with graph_cols[1]:
                hourly_fig = create_hourly_prediction_graph(processed_df, predictions,
                                                            selected_date, selected_brand)
                st.plotly_chart(hourly_fig, use_container_width=True)

            with graph_cols[2]:
                interpolated_hourly_fig = create_interpolated_hourly_chart(
                    processed_df, selected_brand, selected_date
                )
                st.plotly_chart(interpolated_hourly_fig, use_container_width=True)

        else:
            st.warning(f"No data available for {selected_brand}. Please upload data.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure your CSV file contains at least two columns: a date/time column and a value column.")


if __name__ == "__main__":
    main()
