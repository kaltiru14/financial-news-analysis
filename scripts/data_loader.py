import pandas as pd
import os

def load_financial_news(data_path):
    """
    Load financial news data from CSV file
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Basic preprocessing for financial news data
    """
    # Create copy to avoid warnings
    df_processed = df.copy()
    
    # Convert date column
    df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
    
    # Extract features from date
    df_processed['publication_date'] = df_processed['date'].dt.date
    df_processed['publication_hour'] = df_processed['date'].dt.hour
    df_processed['publication_day'] = df_processed['date'].dt.day_name()
    
    # Text features
    df_processed['headline_length'] = df_processed['headline'].astype(str).apply(len)
    df_processed['word_count'] = df_processed['headline'].astype(str).apply(lambda x: len(x.split()))
    
    return df_processed

if __name__ == "__main__":
    # Test the data loader
    data_path = "../data/raw_analyst_ratings.csv"
    df = load_financial_news(data_path)
    if df is not None:
        df_processed = preprocess_data(df)
        print("Data preprocessing completed!")