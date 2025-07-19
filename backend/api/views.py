from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def load_pretrained_model(city_name):
    """Load pre-trained Prophet model for a city"""
    # Normalize city name to match model filename
    city_key = city_name.lower().replace(' ', '')
    model_filename = f"prophet_model_{city_key}.pkl"
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_filename)
    
    if not os.path.exists(model_path):
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model for {city_name}: {str(e)}")
        return None

def get_aqi_category(aqi):
    """Get AQI category based on value"""
    if aqi is None:
        return 'Unknown'
    elif aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

def predict_city_aqi(city_name, target_date):
    """Predict AQI for a city using pre-trained model"""
    model = load_pretrained_model(city_name)
    
    if model is None:
        return None, f"Model not found for city '{city_name}'"
    
    try:
        # Create prediction dataframe
        future_df = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
        
        # Make prediction
        forecast = model.predict(future_df)
        predicted_aqi = round(max(0, forecast['yhat'].values[0]), 2)
        category = get_aqi_category(predicted_aqi)
        
        return predicted_aqi, category
    except Exception as e:
        logger.error(f"Error predicting for {city_name}: {str(e)}")
        return None, str(e)

def get_available_cities():
    """Get list of cities from model files"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    if not os.path.exists(model_dir):
        return []
    
    cities = []
    for filename in os.listdir(model_dir):
        if filename.startswith('prophet_model_') and filename.endswith('.pkl'):
            city_name = filename.replace('prophet_model_', '').replace('.pkl', '').replace('_', ' ').title()
            cities.append(city_name)
    
    return sorted(cities)

@csrf_exempt
@require_http_methods(["POST"])
def predict_aqi(request):
    """API endpoint to predict AQI"""
    try:
        data = json.loads(request.body)
        city = data.get('city', '').strip()
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        if not city:
            return JsonResponse({'error': 'City name required', 'status': 'error'}, status=400)
        
        # Generate 7-day predictions
        predictions = []
        start_date = datetime.strptime(date, '%Y-%m-%d')

        for i in range(-3, 4):
            pred_date = start_date + timedelta(days=i)
            pred_date_str = pred_date.strftime('%Y-%m-%d')
            
            predicted_aqi, category = predict_city_aqi(city, pred_date_str)
            
            predictions.append({
                'date': pred_date_str,
                'aqi': predicted_aqi,
                'category': category if predicted_aqi else 'Unknown'
            })
        
        return JsonResponse({
            'city': city,
            'predictions': predictions,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JsonResponse({'error': str(e), 'status': 'error'}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'message': 'AQI API running',
        'timestamp': datetime.now().isoformat()
    })

@csrf_exempt
@require_http_methods(["GET"])
def get_cities(request):
    """Get available cities"""
    try:
        cities = get_available_cities()
        return JsonResponse({
            'cities': cities,
            'status': 'success',
            'count': len(cities)
        })
    except Exception as e:
        logger.error(f"Error getting cities: {str(e)}")
        return JsonResponse({'error': str(e), 'status': 'error'}, status=500)
