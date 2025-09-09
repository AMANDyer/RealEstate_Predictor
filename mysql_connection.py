from mysql.connector import pooling #for connection to be reuasble 
import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime

# Database config
dbconfig = {
    "host":"localhost",
    "user": "root",
    "password": "1234",
    "database": "indian_real_estate_investment"
}

# Create connection pool
pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **dbconfig)

def get_connection():
    """Get a connection from the pool"""
    return pool.get_connection()

def store_prediction(property_id, prediction_type, actual_value, predicted_value, 
                    prediction_label=None, confidence_score=None, model_version="v1.0", features=None):
    """
    Store model prediction in the database
    
    Args:
        property_id: ID of the property
        prediction_type: 'price', 'rent', or 'label'
        actual_value: Actual target value (if available)
        predicted_value: Model prediction
        prediction_label: For classification models
        confidence_score: Prediction confidence score
        model_version: Version of the model
        features: Dictionary of input features
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Convert features to JSON string if provided
        features_json = json.dumps(features) if features else None
        
        query = """
        INSERT INTO model_predictions 
        (property_id, prediction_type, actual_value, predicted_value, 
         prediction_label, confidence_score, model_version, features_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            property_id, prediction_type, actual_value, predicted_value,
            prediction_label, confidence_score, model_version, features_json
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        print(f"Prediction stored successfully with ID: {prediction_id}")
        return prediction_id
        
    except Exception as e:
        print(f"Error storing prediction: {str(e)}")
        if conn:
            conn.rollback()
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_recent_predictions(limit=10, prediction_type=None):
    """Retrieve recent predictions from the database"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        if prediction_type:
            query = "SELECT * FROM model_predictions WHERE prediction_type = %s ORDER BY prediction_timestamp DESC LIMIT %s"
            cursor.execute(query, (prediction_type, limit))
        else:
            query = "SELECT * FROM model_predictions ORDER BY prediction_timestamp DESC LIMIT %s"
            cursor.execute(query, (limit,))
        
        results = cursor.fetchall()
        return results
        
    except Exception as e:
        print(f"Error retrieving predictions: {str(e)}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Main data processing function
def process_real_estate_data():
    """Main function to process real estate data and fetch weather information"""
    conn = None
    cursor = None
    try:
        # Get connection and cursor
        conn = get_connection()
        cursor = conn.cursor()

        # Execute query
        cursor.execute("""SELECT * FROM indian_real_estates;""")

        # Fetch data and column names
        rows = cursor.fetchall()
        columns = cursor.column_names

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Display the DataFrame
        print(f"Loaded {len(df)} records from database")
        print(df.head())

        #unique cities to get weather details
        unique_cities=df['City'].unique().tolist()
        print(f"Unique cities found: {unique_cities}")

        weather_dict = {}
        API_KEY="55937e877ae6a0d15fa2a65dddb6dc64"

        for city in unique_cities:
            # OpenWeatherMap API endpoint with city name and country code
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
            
            try:
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    weather_dict[city] = {
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'description': data['weather'][0]['description'],
                        'feels_like': data['main']['feels_like']
                    }
                    print(f" Success for {city}: {data['main']['temp']}°C")
                else:
                    print(f"Failed for {city}. Error {response.status_code}: {response.text}")
                    weather_dict[city] = None
                    
            except Exception as e:
                print(f" Exception for {city}: {str(e)}")
                weather_dict[city] = None
            
            time.sleep(1)  # Be polite to the API

        print("\n=== Weather Results ===")
        for city, data in weather_dict.items():
            if data:
                print(f"{city}: {data['temperature']}°C, {data['description']}")

        #Create a temporary DataFrame from weather_dict
        weather_df = pd.DataFrame.from_dict(weather_dict, orient='index')
        weather_df.reset_index(inplace=True)
        weather_df.rename(columns={'index': 'City'}, inplace=True)

        # Now merge with your main DataFrame
        df = df.merge(weather_df, on='City', how='left')

        # Feature engineering
        # 1. Rent_per_Sqft (for price prediction and investment analysis)
        df['Rent_per_Sqft'] = df['Rent'] / df['Size']

        # 2. Log_Rent (for price prediction, handling skewed rent distribution)
        df['Log_Rent'] = np.log1p(df['Rent'])  # log1p handles zero/near-zero values

        # 3. Is_Furnished (binary encoding for furnishing status)
        df['Is_Furnished'] = df['Furnishing Status'].apply(lambda x: 1 if x in ['Furnished', 'Semi-Furnished'] else 0)

        # 4. Monthly_Rental_Income (direct copy of Rent for clarity)
        df['Monthly_Rental_Income'] = df['Rent']

        # 5. Estimated Property Value and Rent_Yield_Estimate
        # City-specific price per sqft (approximated from web data, 2025)
        price_per_sqft = {
            'Mumbai': 20000,
            'Bangalore': 10000,
            'Delhi': 12000,
            'Chennai': 7000,
            'Hyderabad': 8000,
            'Kolkata': 6000
        }
        df['Est_Property_Value'] = df.apply(lambda x: x['Size'] * price_per_sqft.get(x['City'], 10000), axis=1)  # Default 10000 if city not listed
        df['Rent_Yield_Estimate'] = (df['Monthly_Rental_Income'] * 12) / df['Est_Property_Value'] * 100  # Annual yield %

        # 6. Investment_Label (good/bad based on yield > 4% or Rent_per_Sqft above city median)
        # Compute city-wise median Rent_per_Sqft
        city_medians = df.groupby('City')['Rent_per_Sqft'].median()
        df['City_Median_Rent_Sqft'] = df['City'].map(city_medians)
        df['Investment_Label'] = ((df['Rent_Yield_Estimate'] > 4) | (df['Rent_per_Sqft'] > df['City_Median_Rent_Sqft'])).astype(int)

        # 7. Locality_Desirability_Score (based on Rent_per_Sqft percentile within each city)
        df['Locality_Desirability_Score'] = df.groupby('City')['Rent_per_Sqft'].transform(
            lambda x: pd.qcut(x, q=10, labels=range(1, 11), duplicates='drop')
        ).astype(float)

        # 8. City_Encoded (label encoding for City)
        city_mapping = {city: idx for idx, city in enumerate(df['City'].unique())}
        df['City_Encoded'] = df['City'].map(city_mapping)

        # Save the updated dataset
        df.to_csv('indian_real_estate.csv', index=False)
        print(f"Dataset saved to 'indian_real_estate.csv' with {len(df)} records")

        # Display sample of new columns
        print("\n=== Sample of Processed Data ===")
        print(df[['Rent', 'Size', 'City', 'Rent_per_Sqft', 'Log_Rent', 'Is_Furnished', 
                'Monthly_Rental_Income', 'Est_Property_Value', 'Rent_Yield_Estimate', 
                'Investment_Label', 'Locality_Desirability_Score', 'City_Encoded']].head())
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None
    finally:
        # Properly close resources
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    # Run the data processing when script is executed directly
    process_real_estate_data()
    
    