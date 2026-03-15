import joblib
import numpy as np
import pandas as pd

print("=" * 80)
print("TESTING RANDOM FOREST MODELS")
print("=" * 80)

# List of models to test
models = {
    'Delhi': 'Delhi_random_forest_model.pkl',
    'GNoida': 'GNoida_random_forest_model.pkl',
    'Ghaziabad': 'Ghaziabad_random_forest_model.pkl',
    'Lucknow': 'Lucknow_random_forest_model.pkl'
}

# Load imputer
try:
    imputer = joblib.load('imputer.pkl')
    
    # Fix for SimpleImputer compatibility with scikit-learn 1.8.0+
    if not hasattr(imputer, '_fill_dtype'):
        import numpy as np
        if hasattr(imputer, 'statistics_'):
            setattr(imputer, '_fill_dtype', imputer.statistics_.dtype)
        else:
            setattr(imputer, '_fill_dtype', np.float64)
    
    print(f"\n✓ Loaded imputer successfully (with compatibility fix)")
except Exception as e:
    print(f"\n❌ Error loading imputer: {e}")
    exit(1)

# Test data (sample pollutant values)
test_data = pd.DataFrame({
    'pm25': [50.5],
    'pm10': [80.3],
    'o3': [40.2],
    'no2': [35.7],
    'so2': [12.4],
    'co': [1.2]
})

print(f"\nTest Input Data:")
print(test_data)
print()

# Test each model
for city_name, model_file in models.items():
    print(f"\n{'-' * 80}")
    print(f"Testing {city_name} Model: {model_file}")
    print(f"{'-' * 80}")
    
    try:
        # Load model
        model = joblib.load(model_file)
        
        # Fix for tree estimators compatibility
        estimators = getattr(model, 'estimators_', None)
        if estimators is not None:
            for est in estimators:
                if not hasattr(est, 'monotonic_cst'):
                    setattr(est, 'monotonic_cst', None)
        if not hasattr(model, 'monotonic_cst'):
            setattr(model, 'monotonic_cst', None)
        
        print(f"✓ Model loaded successfully")
        
        # Apply imputer
        test_imputed = imputer.transform(test_data)
        print(f"✓ Data imputed successfully")
        
        # Make prediction
        prediction = model.predict(test_imputed)
        print(f"✓ Prediction successful")
        print(f"  Predicted AQI: {prediction[0]:.2f}")
        
        # Check model attributes
        if hasattr(model, 'n_estimators'):
            print(f"  Number of trees: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"  Max depth: {model.max_depth}")
        
        print(f"✅ {city_name} model is working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing {city_name} model: {str(e)}")

print(f"\n{'=' * 80}")
print("MODEL TESTING COMPLETED")
print(f"{'=' * 80}\n")
