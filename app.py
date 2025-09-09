from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from mysql_connection import store_prediction, get_recent_predictions
import json
from datetime import datetime
from flask_cors import CORS
from flask import render_template  # Add this import


# Initialize the Flask app
app = Flask(__name__)
CORS(app)


# Load the models
model1_path = "investment_Price.joblib"
model2_path = "rental_income.joblib"
model3_path = "investment_label.joblib"

try:
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    model3 = joblib.load(model3_path)
    print("All models loaded successfully")
except Exception as e:
    raise Exception(f"Error loading models: {str(e)}")

# Define the prediction endpoint with storage
@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert the input data to a Pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure the columns are in the correct order as per training
        columns = [
            'BHK', 'Size', 'Area Type', 'Area Locality', 'City', 'Furnishing Status',
            'Tenant Preferred', 'Bathroom', 'Point of Contact', 'temperature',
            'humidity', 'Rent_per_Sqft', 'City_Median_Rent_Sqft',
            'Locality_Desirability_Score', 'Current_Floor', 'Total_Floors',
            'Monthly_Rental_Income'
        ]
        
        # Reorder columns to match expected order
        input_df = input_df[columns]
        
        # Make the prediction using the loaded model
        prediction = model1.predict(input_df)
        predicted_value = float(prediction[0])
        
        # Store prediction in database
        prediction_id = store_prediction(
            property_id=data.get('property_id', 'unknown'),
            prediction_type='price',
            actual_value=data.get('Est_Property_Value'),  # If available in request
            predicted_value=predicted_value,
            model_version="investment_price_v1",
            features=data
        )
        
        # Return the prediction as JSON with additional metadata
        return jsonify({
            'prediction': predicted_value,
            'prediction_id': prediction_id,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': 'Price prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e), 
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/predict_rent', methods=['POST'])
def predict_rent():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert the input data to a Pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure the columns are in the correct order as per training
        columns = [
            "BHK", "Size", "Area Type", "Area Locality", "City",
            "Furnishing Status", "Tenant Preferred", "Bathroom",
            "Point of Contact", "temperature", "humidity", "Rent_per_Sqft",
            "City_Median_Rent_Sqft", "Locality_Desirability_Score",
            "Current_Floor", "Total_Floors"
        ]
        
        # Reorder columns to match expected order
        input_df = input_df[columns]
        
        # Make the prediction using the loaded model
        prediction = model2.predict(input_df)
        predicted_value = float(prediction[0])
        
        # Store prediction in database
        prediction_id = store_prediction(
            property_id=data.get('property_id', 'unknown'),
            prediction_type='rent',
            actual_value=data.get('Monthly_Rental_Income'),  # If available
            predicted_value=predicted_value,
            model_version="rental_income_v1",
            features=data
        )
        
        # Return the prediction as JSON
        return jsonify({
            'prediction': predicted_value,
            'prediction_id': prediction_id,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': 'Rental income prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e), 
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/predict_label', methods=['POST'])
def predict_label():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert the input data to a Pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure the columns are in the correct order as per training
        columns = [
            "BHK", "Size", "Area Type", "Area Locality", "City",
            "Furnishing Status", "Tenant Preferred", "Bathroom",
            "Point of Contact", "temperature", "humidity", "Rent_per_Sqft",
            "City_Median_Rent_Sqft", "Locality_Desirability_Score",
            "Current_Floor", "Total_Floors", "Rent_Yield_Estimate",
            "Est_Property_Value"
        ]
        
        # Reorder columns to match expected order
        input_df = input_df[columns]
        
        # Make the prediction using the loaded model
        prediction = model3.predict(input_df)
        predicted_label = str(prediction[0])
        
        # Get prediction probabilities for confidence score
        try:
            probabilities = model3.predict_proba(input_df)
            confidence_score = float(max(probabilities[0]))
        except:
            confidence_score = None
        
        # Store prediction in database
        prediction_id = store_prediction(
            property_id=data.get('property_id', 'unknown'),
            prediction_type='label',
            actual_value=data.get('Investment_Label'),  # If available
            predicted_value=None,  # For classification, use prediction_label
            prediction_label=predicted_label,
            confidence_score=confidence_score,
            model_version="investment_label_v1",
            features=data
        )
        
        # Return the prediction as JSON
        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence_score,
            'prediction_id': prediction_id,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': 'Investment label prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e), 
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 400

# New endpoint to retrieve prediction history
@app.route('/prediction_history', methods=['GET'])
def prediction_history():
    try:
        # Get query parameters
        prediction_type = request.args.get('type')
        limit = int(request.args.get('limit', 10))
        property_id = request.args.get('property_id')
        
        # Get predictions from database
        predictions = get_recent_predictions(limit, prediction_type)
        
        # Filter by property_id if provided
        if property_id:
            predictions = [p for p in predictions if p.get('property_id') == property_id]
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e), 
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': True,
        'message': 'Flask app is running successfully'
    })

# Model information endpoint
@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'models': {
            'investment_price': {
                'type': 'regression',
                'target': 'Est_Property_Value',
                'version': 'v1.0'
            },
            'rental_income': {
                'type': 'regression', 
                'target': 'Monthly_Rental_Income',
                'version': 'v1.0'
            },
            'investment_label': {
                'type': 'classification',
                'target': 'Investment_Label',
                'version': 'v1.0'
            }
        },
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })


# Run the app
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available endpoints:")
    print("  POST /predict_price    - Predict property price")
    print("  POST /predict_rent     - Predict rental income") 
    print("  POST /predict_label    - Predict investment label")
    print("  GET  /prediction_history - Get prediction history")
    print("  GET  /health           - Health check")
    print("  GET  /model_info       - Model information")
    
    app.run(debug=F, host='0.0.0.0', port=5000)