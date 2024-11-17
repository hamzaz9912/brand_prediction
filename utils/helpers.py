import pandas as pd
import os

def save_uploaded_file(uploaded_file, brand, data_dir):
    """Save uploaded file to the data directory"""
    try:
        if uploaded_file is not None:
            file_path = os.path.join(data_dir, f"{brand.lower()}.csv")
            df = pd.read_csv(uploaded_file)
            df.to_csv(file_path, index=False)
            return df
        return None
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")

def load_brand_data(brand, data_dir):
    """Load existing brand data"""
    try:
        file_path = os.path.join(data_dir, f"{brand.lower()}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
