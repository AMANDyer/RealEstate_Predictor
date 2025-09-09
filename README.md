# RealEstateInsight - Property Investment Predictor

## 📋 Project Overview

RealEstateInsight is a comprehensive machine learning-powered web application that provides accurate predictions for property values, rental income, and investment potential. The system leverages multiple XGBoost models trained on Indian real estate data to deliver intelligent investment insights.

## 🌟 Key Features

### 1. Weather API Integration for Enhanced Predictions
- **Purpose**: Fetches real-time weather data for each property location
- **Implementation**: Integrates with OpenWeatherMap API to gather temperature and humidity data
- **Impact**: Weather conditions significantly influence property desirability and rental patterns, making this data valuable for accurate predictions

### 2. Multi-Model Prediction System
- **Property Value Prediction**: Estimates market price using XGBoost regression
- **Rental Income Prediction**: Projects monthly rental income with regression modeling
- **Investment Classification**: Classifies properties as "Excellent" or "Bad" investments using XGBoost classification

### 3. Comprehensive Data Processing Pipeline
- **Data Collection**: Extracts property data from MySQL database
- **Feature Engineering**: Creates meaningful features like:
  - Rent per square foot
  - City-specific price metrics
  - Rental yield estimates
  - Locality desirability scores
- **Weather Integration**: Enhances dataset with climate information

### 4. Intelligent Feature Engineering
- **Floor Information Extraction**: Parses complex floor descriptions into usable numerical data
- **Skew Handling**: Applies logarithmic transformations to handle skewed target variables
- **Categorical Encoding**: Uses one-hot encoding for categorical features
- **Location Scoring**: Creates desirability scores based on rental price percentiles

### 5. Web Application Interface
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **User-Friendly Forms**: Multi-step form for easy data entry
- **Real-Time Results**: Instant prediction display with visual indicators
- **Prediction History**: Tracks and displays previous predictions

### 6. Database Integration
- **MySQL Connection Pooling**: Efficient database connections for high performance
- **Prediction Storage**: Saves all predictions with timestamps and feature data
- **History Retrieval**: Easy access to past predictions for analysis

### 7. Model Optimization
- **Hyperparameter Tuning**: Uses RandomizedSearchCV for optimal model performance
- **Cross-Validation**: Ensures model robustness with k-fold validation
- **Performance Metrics**: Tracks RMSE, MAE, R² for regression; accuracy, F1-score for classification

## 🛠️ Technical Architecture

### Backend Components:
- **Flask API**: RESTful endpoints for predictions
- **XGBoost Models**: Three specialized machine learning models
- **MySQL Database**: Data storage and prediction history
- **Joblib**: Model serialization and loading

### Frontend Components:
- **HTML5/CSS3**: Responsive and modern UI design
- **JavaScript**: Dynamic form handling and API communication
- **Chart.js**: Data visualization for results

### Data Processing:
- **Pandas/NumPy**: Data manipulation and transformation
- **Scikit-learn**: Preprocessing and model evaluation
- **Weather API**: External data integration

## 📊 Model Performance

### Regression Models (Price & Rental Income):
- **RMSE**: Low error rates on test data
- **R² Score**: High explanatory power
- **Log Transformation**: Handles skewed distribution effectively

### Classification Model (Investment Label):
- **Accuracy**: High prediction accuracy
- **F1-Score**: Balanced precision and recall
- **Confidence Scores**: Probability estimates for predictions

## 🚀 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Database Setup**:
   - Configure MySQL connection in `mysql_connection.py`
   - Ensure the database schema matches expected structure

3. **API Key Setup**:
   - Obtain OpenWeatherMap API key
   - Replace placeholder in `mysql_connection.py`

4. **Run Application**:
   ```bash
   python app.py
   ```

5. **Access Web Interface**:
   Open `http://localhost:5000` in your browser

## 📈 Usage Flow

1. **Data Collection**: System gathers property data and weather information
2. **Feature Engineering**: Creates enhanced features for modeling
3. **Model Training**: Trains three specialized XGBoost models
4. **Prediction**: User inputs property details through web form
5. **Results Display**: System shows price estimate, rental income, and investment potential
6. **History Storage**: Prediction saved to database for future reference

## 🔮 Future Enhancements

- Integration with real property listing APIs
- Advanced visualization of market trends
- Mobile application development
- Additional factors like neighborhood amenities and transportation
- Time-series analysis for price forecasting

## 📝 Conclusion

RealEstateInsight demonstrates a complete machine learning pipeline from data acquisition through weather API integration to deployment as a web application. The system provides valuable insights for property investors by combining traditional real estate metrics with environmental factors and advanced machine learning techniques.
