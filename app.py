import os
import numpy as np
import pandas as pd
import re
import joblib
from flask import Flask, render_template, request, jsonify
import warnings

# Initialize Flask app
app = Flask(__name__)

# Global variables to store models and preprocessing tools
models = {}
encoders = {}
scalers = {}

# Suppress warnings
warnings.filterwarnings("ignore")


# Function to load all necessary models and preprocessing tools
def load_models_and_tools():
    global models, encoders, scalers

    print("Loading encoders and scalers...")
    try:
        # Load encoders
        encoders = joblib.load('utils/encoders.pkl')
        print("Encoders loaded successfully")

        # Load scalers
        scalers = joblib.load('utils/scalers.pkl')
        print("Scalers loaded successfully")
    except Exception as e:
        print(f"Error loading encoders/scalers: {str(e)}")
        # Initialize empty dictionaries if loading fails
        if not encoders:
            encoders = {}
        if not scalers:
            scalers = {}

    # We'll use a more resilient approach for model loading
    print("Loading models...")
    try:
        # Try to load the RandomForest models
        rf_models = {
            'lstm_rf': 'models/lstm_rf_rf.pkl',
            'cnn_rf': 'models/cnn_rf_rf.pkl',
            'rnn_rf': 'models/rnn_rf_rf.pkl'
        }

        # Load each RF model
        for model_name, model_path in rf_models.items():
            try:
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model")
                else:
                    print(f"Warning: Model file {model_path} not found")
            except Exception as e:
                print(f"Error loading {model_name} model: {str(e)}")

        # Since there might be TensorFlow compatibility issues, we'll handle Keras models separately
        try:
            # Only import TensorFlow if we need it
            import tensorflow as tf
            from tensorflow.keras.models import load_model

            keras_models = {
                'cnn_rnn': 'models/cnn_rnn_model.h5',
                'lstm': 'models/lstm_model.h5',
                'cnn': 'models/cnn_model.h5',
                'rnn': 'models/rnn_model.h5'
            }

            for model_name, model_path in keras_models.items():
                try:
                    if os.path.exists(model_path):
                        # Try to load with custom_objects for compatibility
                        models[model_name] = load_model(model_path, compile=False)
                        print(f"Loaded {model_name} model")
                    else:
                        print(f"Warning: Model file {model_path} not found")
                except Exception as e:
                    print(f"Error loading {model_name} model: {str(e)}")

        except ImportError:
            print("TensorFlow/Keras not available. Skipping deep learning models.")

    except Exception as e:
        print(f"Error during model loading: {str(e)}")

    # Print summary of loaded models
    print(f"Successfully loaded {len(models)} models")
    print(f"Available models: {list(models.keys())}")


# Function to preprocess user input
def preprocess_input(user_input):
    """Process user input to match the format needed for prediction"""
    # Extract input values
    brand = user_input.get('brand')
    car_model = user_input.get('car_model')
    model_year = int(user_input.get('model_year'))
    kilometers_run = float(user_input.get('kilometers_run'))
    engine_capacity = user_input.get('engine_capacity')
    transmission = user_input.get('transmission')
    body_type = user_input.get('body_type')
    fuel_type = user_input.get('fuel_type')

    # Create a DataFrame with a single row
    input_df = pd.DataFrame({
        'brand': [brand],
        'car_model': [car_model],
        'model_year': [model_year],
        'kilometers_run': [kilometers_run],
        'Engine_capacity': [engine_capacity],
        'transmission': [transmission],
        'body_type': [body_type],
        'fuel_type': [fuel_type]
    })

    # Extract engine capacity numeric value
    input_df['engine_capacity_numeric'] = input_df['Engine_capacity'].apply(
        lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if isinstance(x, str) and re.search(r'(\d+)', str(x)) else 0
    )

    # Calculate car age
    current_year = 2025
    input_df['car_age'] = current_year - input_df['model_year']

    # Log transform kilometers
    input_df['log_kilometers'] = np.log1p(input_df['kilometers_run'])

    # Normalize transmission
    input_df['transmission'] = input_df['transmission'].apply(
        lambda x: 'Automatic' if str(x).lower() in ['auto', 'automatic'] else x
    )

    # Encode categorical features
    for col in ['brand', 'car_model', 'transmission', 'body_type', 'fuel_type']:
        # Create encoded column using the saved encoder
        encoder = encoders.get(col, {})
        # If the value is not in the encoder, use -1 (unknown)
        input_df[f'{col}_encoded'] = input_df[col].map(lambda x: encoder.get(x, -1))

    # Select features in the correct order
    numeric_features = [
        'model_year', 'kilometers_run', 'engine_capacity_numeric', 'car_age', 'log_kilometers'
    ]

    categorical_features = [
        'brand_encoded', 'car_model_encoded', 'transmission_encoded',
        'body_type_encoded', 'fuel_type_encoded'
    ]

    selected_features = numeric_features + categorical_features

    # Extract features
    features = input_df[selected_features].values

    # Scale features if scaler is available
    if 'feature_scaler' in scalers:
        feature_scaler = scalers['feature_scaler']
        features_scaled = feature_scaler.transform(features)
    else:
        # If scaler is not available, just use the unscaled features
        print("Warning: Feature scaler not available. Using unscaled features.")
        features_scaled = features

    # Reshape for deep learning models
    features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)

    return features_scaled, features_reshaped


# Function to make predictions using available models
def predict_price(features_scaled, features_reshaped, model_name='ensemble'):
    """Make a prediction using the specified model or ensemble of available models"""

    # Check if any models are available
    if not models:
        return 500000  # Return a default value if no models are available

    # Check if the requested model is available
    if model_name != 'ensemble' and model_name not in models:
        print(f"Warning: Requested model '{model_name}' not available. Using ensemble instead.")
        model_name = 'ensemble'

    # If we're using the ensemble but have no models, return a default value
    if model_name == 'ensemble' and not models:
        return 500000

    try:
        # Get target transformer if available
        target_transformer = scalers.get('target_transformer')

        # For CNN+RNN direct model
        if model_name == 'cnn_rnn' and 'cnn_rnn' in models:
            # Predict with CNN+RNN model
            import tensorflow as tf
            prediction_transformed = models['cnn_rnn'].predict(features_reshaped, verbose=0).flatten()

            # Transform back if transformer is available
            if target_transformer:
                prediction = target_transformer.inverse_transform(prediction_transformed.reshape(-1, 1)).flatten()[0]
            else:
                prediction = prediction_transformed[0]

        # For hybrid models (LSTM+RF, CNN+RF, RNN+RF)
        elif model_name in ['lstm_rf', 'cnn_rf', 'rnn_rf'] and model_name in models:
            # For hybrid models, we'll just use the RF part directly
            prediction = models[model_name].predict(features_scaled)[0]

        # For ensemble
        elif model_name == 'ensemble':
            predictions = []

            # Try each available model
            for model_key, model in models.items():
                try:
                    if model_key == 'cnn_rnn':
                        # Deep learning model needs special handling
                        import tensorflow as tf
                        pred_transformed = model.predict(features_reshaped, verbose=0).flatten()
                        if target_transformer:
                            pred = target_transformer.inverse_transform(pred_transformed.reshape(-1, 1)).flatten()[0]
                        else:
                            pred = pred_transformed[0]
                        predictions.append(pred)

                    elif model_key in ['lstm_rf', 'cnn_rf', 'rnn_rf']:
                        # RandomForest models
                        pred = model.predict(features_scaled)[0]
                        predictions.append(pred)

                except Exception as e:
                    print(f"Error predicting with {model_key}: {str(e)}")

            # If we have predictions, return the average
            if predictions:
                prediction = np.mean(predictions)
            else:
                prediction = 500000  # Default if all models fail

        else:
            # Should not reach here normally, but just in case
            prediction = 500000

        return prediction

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 500000  # Return a default value in case of error


# Routes
@app.route('/')
def home():
    # Get unique values from encoders for dropdown menus
    brands = list(encoders.get('brand', {}).keys())
    car_models = list(encoders.get('car_model', {}).keys())
    transmissions = list(encoders.get('transmission', {}).keys())
    body_types = list(encoders.get('body_type', {}).keys())
    fuel_types = list(encoders.get('fuel_type', {}).keys())

    # Default options if encoders are not available
    if not brands:
        brands = ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra"]
    if not car_models:
        car_models = ["Swift", "i20", "City", "Corolla", "Nexon", "XUV500"]
    if not transmissions:
        transmissions = ["Manual", "Automatic"]
    if not body_types:
        body_types = ["Hatchback", "Sedan", "SUV", "MUV"]
    if not fuel_types:
        fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]

    # List available models for the form
    available_models = list(models.keys())

    return render_template('index.html',
                           brands=brands,
                           car_models=car_models,
                           transmissions=transmissions,
                           body_types=body_types,
                           fuel_types=fuel_types,
                           available_models=available_models)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        user_input = {
            'brand': request.form.get('brand'),
            'car_model': request.form.get('car_model'),
            'model_year': request.form.get('model_year'),
            'kilometers_run': request.form.get('kilometers_run'),
            'engine_capacity': request.form.get('engine_capacity'),
            'transmission': request.form.get('transmission'),
            'body_type': request.form.get('body_type'),
            'fuel_type': request.form.get('fuel_type')
        }

        # Get selected model
        model_name = request.form.get('model', 'ensemble')

        # Preprocess input
        features_scaled, features_reshaped = preprocess_input(user_input)

        # Make prediction
        prediction = predict_price(features_scaled, features_reshaped, model_name)

        # Format prediction
        formatted_prediction = format(prediction, ',.2f')

        return render_template('result.html',
                               prediction=formatted_prediction,
                               car_details=user_input,
                               model_used=model_name)

    except Exception as e:
        error_message = str(e)
        print(f"Error during prediction: {error_message}")
        return render_template('error.html', error=error_message)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Get selected model
        model_name = data.get('model', 'ensemble')

        # Preprocess input
        features_scaled, features_reshaped = preprocess_input(data)

        # Make prediction
        prediction = predict_price(features_scaled, features_reshaped, model_name)

        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': format(prediction, ',.2f'),
            'model_used': model_name
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_car_models/<brand>')
def get_car_models(brand):
    """Get car models for the selected brand"""
    # This would typically filter car models by brand from a database
    # For simplicity, we'll return all car models for now
    car_models = list(encoders.get('car_model', {}).keys())
    return jsonify(car_models)


# Ensure directories exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created models directory")

if not os.path.exists('utils'):
    os.makedirs('utils')
    print("Created utils directory")

if not os.path.exists('templates'):
    os.makedirs('templates')
    print("Created templates directory")


# Create a function to initialize models at startup
def init_app(app):
    with app.app_context():
        load_models_and_tools()


if __name__ == '__main__':
    # Initialize models before starting the app
    print("Initializing app...")
    init_app(app)

    # Start the app
    print("Starting Flask server...")
    app.run(debug=True)