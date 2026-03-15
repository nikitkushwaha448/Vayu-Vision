import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

MIN_SAMPLES_REQUIRED = 30

# Define cities and their corresponding CSV files
cities = {
    'Delhi': 'r.k.-puram, delhi, delhi, india-air-quality.csv',
    'GNoida': 'knowledge-park - iii, greater noida, india-air-quality.csv',
    'Ghaziabad': 'sanjay-nagar, ghaziabad, india-air-quality.csv',
    'Lucknow': 'talkatora,-lucknow, india-air-quality.csv',
    'Hyderabad': 'hyderabad-air-quality.csv',
    'Mumbai': 'mumbai-air-quality.csv',
    'Ahmedabad': 'ahmedabad-air-quality.csv',
    'Punjab': 'punjab-air-quality.csv',
    'Gurgaon': 'gurgaon-air-quality.csv',
    'Chennai': 'chennai-air-quality.csv',
    'Kerala': 'kerala-air-quality.csv',
    'Nagaland': 'nagaland-air-quality.csv'
}

# Note: Add more cities when you have their CSV data files
# 'Lucknow': 'talkatora,-lucknow, india-air-quality.csv',
# 'Noida': 'noida-air-quality.csv',

# Define the input parameters and the target variable
input_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
output_col = 'AQI'

print("=" * 80)
print("RANDOM FOREST MODEL TRAINING FOR MULTIPLE CITIES")
print("=" * 80)

# Train models for each city
for city_name, csv_file in cities.items():
    print(f"\n{'=' * 80}")
    print(f"Training model for: {city_name}")
    print(f"Data file: {csv_file}")
    print(f"{'=' * 80}")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"⚠️  WARNING: File '{csv_file}' not found. Skipping {city_name}...")
        continue
    
    try:
        # Load the dataset
        data = pd.read_csv(csv_file)
        print(f"✓ Loaded dataset with {len(data)} rows")
        
        # Replace empty spaces with NaN
        data.replace(' ', pd.NA, inplace=True)
        data.replace('', pd.NA, inplace=True)
        
        # Check if required columns exist
        missing_cols = [col for col in input_cols + [output_col] if col not in data.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing columns {missing_cols}. Skipping {city_name}...")
            continue
        
        # Extract input features and target variable
        X = data[input_cols].copy()
        y = data[output_col]
        
        # Convert to numeric, handling any non-numeric values
        for col in input_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove rows where AQI is missing
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"✓ Valid samples after removing missing AQI: {len(y)}")
        
        if len(y) < MIN_SAMPLES_REQUIRED:
            print(f"⚠️  WARNING: Not enough valid samples ({len(y)}). Skipping {city_name}...")
            continue
        
        # Impute missing values with the mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        print(f"✓ Imputed missing values in features")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Split data - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Initialize and fit the RandomForestRegressor model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        print(f"✓ Trained Random Forest model")
        
        # Evaluate the model
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   - Mean Squared Error: {mse:.2f}")
        print(f"   - R² Score: {r2:.4f}")
        
        # Save the trained model
        model_filename = f'{city_name}_random_forest_model.pkl'
        joblib.dump(rf_model, model_filename)
        print(f"✓ Saved model to: {model_filename}")
        
        # Save the imputer (one imputer for all cities)
        if not os.path.exists('imputer.pkl'):
            imputer_filename = 'imputer.pkl'
            joblib.dump(imputer, imputer_filename)
            print(f"✓ Saved imputer to: {imputer_filename}")
        
        print(f"✅ Successfully completed training for {city_name}")
        
    except Exception as e:
        print(f"❌ ERROR training model for {city_name}: {str(e)}")
        continue

print(f"\n{'=' * 80}")
print("MODEL TRAINING COMPLETED")
print(f"{'=' * 80}\n")

# List all generated model files
model_files = [f for f in os.listdir('.') if f.endswith('_random_forest_model.pkl')]
print(f"Generated models ({len(model_files)}):")
for model_file in model_files:
    file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
    print(f"  - {model_file} ({file_size:.2f} MB)")

if os.path.exists('imputer.pkl'):
    imputer_size = os.path.getsize('imputer.pkl') / 1024  # Size in KB
    print(f"  - imputer.pkl ({imputer_size:.2f} KB)")

print("\n✅ All models are ready to use!")
