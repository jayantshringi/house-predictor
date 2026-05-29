import os
import numpy as np
import pandas as pd

def generate_synthetic_housing_data(filepath: str, n_samples: int = 1000):
    """Generates a realistic synthetic housing dataset and writes to CSV."""
    np.random.seed(42)
    
    # Generate features
    square_feet = np.random.uniform(800, 4000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    # Bathrooms generally correlate with bedrooms
    bathrooms = np.clip(bedrooms - np.random.randint(0, 2, n_samples), 1, 4)
    age_of_house = np.random.uniform(0, 80, n_samples)
    
    # Calculate price using a linear model with noise
    # Base price: 50,000
    # + 120 per sqft
    # + 25,000 per bedroom
    # + 15,000 per bathroom
    # - 800 per year of age
    noise = np.random.normal(0, 15000, n_samples)
    price = 50000 + 120 * square_feet + 25000 * bedrooms + 15000 * bathrooms - 800 * age_of_house + noise
    
    # Clip price to be positive just in case
    price = np.clip(price, 30000, None)
    
    df = pd.DataFrame({
        'SquareFeet': square_feet.round(0),
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'AgeOfHouse': age_of_house.round(0),
        'Price': price.round(2)
    })
    
    # Inject a few outliers (both in SquareFeet and Price) to test outlier removal
    # 5 extreme outliers
    for i in range(5):
        df.loc[i, 'SquareFeet'] = df.loc[i, 'SquareFeet'] * 5
        df.loc[i, 'Price'] = df.loc[i, 'Price'] * 6
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated {n_samples} housing records with outliers. Saved to {filepath}")

if __name__ == '__main__':
    generate_synthetic_housing_data('data/housing.csv')
