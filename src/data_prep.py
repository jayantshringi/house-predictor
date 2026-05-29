import pandas as pd
import numpy as np

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Loads CSV and ensures no critical structural issues exist."""
    df = pd.read_csv(filepath)
    
    # Target and Feature validation
    required_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'AgeOfHouse', 'Price']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory column: {col}")
            
    # Remove null values or fill them using an explicit strategy
    df = df.dropna(subset=required_columns)
    
    # Outlier removal via Interquartile Range (IQR) for numerical pillars
    for col in ['SquareFeet', 'Price']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df
