import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer

# Load Data
df = pd.read_csv('final_data1.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encode wind direction
df['WindDirection_sin'] = np.sin(np.radians(df['Wind Direction at 2 Meters (Degrees)']))
df['WindDirection_cos'] = np.cos(np.radians(df['Wind Direction at 2 Meters (Degrees)']))

# Create lagged precipitation features
for i in range(1, 31):
    weight = np.exp(-i / 7)
    df[f'Precipitation_lag{i}_weighted'] = df['Precipitation Corrected (mm/day)'].shift(i) * weight

# Create rolling averages
df['Temperature_rolling7'] = df['Temperature at 2 meters (c)'].rolling(window=7).mean()
df['Wind_Speed_rolling7'] = df['Wind Speed at 2 Meters (m/s)'].rolling(window=7).mean()

# Add new interaction features
df['Wind_Precip_Interaction'] = df['Wind Speed at 2 Meters (m/s)'] * df['Precipitation Corrected (mm/day)']
df['Wind_Humidity_Interaction'] = df['Wind Speed at 2 Meters (m/s)'] * df['Specific Humidity at 2 Meters (g/kg)']
df['Log_Humidity'] = np.log1p(df['Specific Humidity at 2 Meters (g/kg)'])

# Drop unnecessary columns
df.drop(columns=['Wind Direction at 2 Meters (Degrees)'], inplace=True)

# Handle missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['Date'])), columns=df.columns[1:])
df_imputed['Date'] = df['Date']

# Define target variables
target_columns = [
    'Temperature at 2 meters (c)', 'Precipitation Corrected (mm/day)',
    'Specific Humidity at 2 Meters (g/kg)', 'Wind Speed at 2 Meters (m/s)',
    'Root Zone Soil Wetness'
]
# Before model training, add seasonal features
# ...
# Add seasonal features to df_imputed instead of df
df_imputed['season_factor'] = np.sin(2 * np.pi * (df_imputed['Month'] + df_imputed['Day']/30) / 12)
df_imputed['temp_factor'] = df_imputed['Temperature at 2 meters (c)'] * df_imputed['season_factor']
df_imputed['humid_factor'] = df_imputed['Specific Humidity at 2 Meters (g/kg)'] * (1 + df_imputed['season_factor'])
df_imputed['wind_factor'] = df_imputed['Wind Speed at 2 Meters (m/s)'] * (1 + 0.5 * df_imputed['season_factor'])
df_imputed['precip_factor'] = df_imputed['Precipitation Corrected (mm/day)'] * (1 + 0.3 * np.cos(2 * np.pi * df_imputed['Month'] / 12))

# Now update features with the new columns
features = df_imputed.drop(columns=['Date'] + target_columns + ['season_factor', 'temp_factor', 'humid_factor', 'wind_factor', 'precip_factor'])

# Update model training parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 100,
    'max_depth': 15,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'num_iterations': 300
}

X_train, X_test, y_train, y_test = train_test_split(features, df_imputed[target_columns], test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LightGBM models
models = {}
for column in target_columns:
    lgb_train = lgb.Dataset(X_train_scaled, y_train[column])
    lgb_eval = lgb.Dataset(X_test_scaled, y_test[column], reference=lgb_train)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'huber' if column == 'Precipitation Corrected (mm/day)' else 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 80 if column in ['Wind Speed at 2 Meters (m/s)', 'Specific Humidity at 2 Meters (g/kg)'] else 50,
        'max_depth': 10,
        'n_estimators': 2000 if column == 'Precipitation Corrected (mm/day)' else 200,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'min_child_samples': 10,
        'reg_alpha': 0.3,
        'reg_lambda': 0.4,
        'verbose': -1
    }
    
    model = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_eval], callbacks=[lgb.early_stopping(stopping_rounds=20)])
    models[column] = model
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test[column], y_pred)
    r2 = r2_score(y_test[column], y_pred)
    print(f"\nResults for {column}:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Prediction function
def predict_weather(input_data):
    input_scaled = scaler.transform(input_data)
    return {col: models[col].predict(input_scaled)[0] for col in target_columns}

# User input function
def get_user_input():
    """Get date input from user and prepare features"""
    while True:
        try:
            print("\nEnter date (DD/MM/YYYY):")
            date_str = input()
            date = pd.to_datetime(date_str, format='%d/%m/%Y')
            
            if date.year < 1900 or date.year > 2100:
                print("Please enter a date between 1900 and 2100")
                continue
            
            # Calculate seasonal factors
            annual_cycle = 2 * np.pi * ((date.month * 30 + date.day) / 365)
            season_factor = np.sin(annual_cycle)
            
            # Create base features with seasonal variations
            base_temp = 20 + 10 * np.sin(annual_cycle - np.pi/6)  # Temperature peaks in summer
            base_wind = 5 + 3 * np.cos(annual_cycle)  # Wind varies seasonally
            base_humid = 8 + 4 * np.sin(annual_cycle)  # Humidity follows temperature
            
            user_data = {
                'Year': date.year,
                'Month': date.month,
                'Day': date.day,
                'WindDirection_sin': np.sin(annual_cycle),
                'WindDirection_cos': np.cos(annual_cycle),
                'Temperature_rolling7': base_temp,
                'Wind_Speed_rolling7': base_wind,
                'Wind_Precip_Interaction': base_wind * max(0, 2 + season_factor),
                'Wind_Humidity_Interaction': base_wind * base_humid,
                'Log_Humidity': np.log1p(base_humid)
            }
            
            # Add precipitation lag features with seasonal variation
            for i in range(1, 31):
                weight = np.exp(-i / 7)
                user_data[f'Precipitation_lag{i}_weighted'] = max(0, 2 + season_factor) * weight
            
            input_features = pd.DataFrame([user_data])
            return input_features[features.columns]
            
        except ValueError:
            print("Invalid date format. Please use DD/MM/YYYY format.")

# Update the main loop to be simpler
def format_prediction(param, value):
    """Format the prediction output with units"""
    units = {
        'Temperature at 2 meters (c)': '°C',
        'Precipitation Corrected (mm/day)': 'mm/day',
        'Specific Humidity at 2 Meters (g/kg)': 'g/kg',
        'Wind Speed at 2 Meters (m/s)': 'm/s',
        'Root Zone Soil Wetness': ''
    }
    return f"{param}: {value:.2f} {units.get(param, '')}"

# Add after model training and before prediction functions
import pickle

# Update the feature engineering section
def save_models():
    # First, ensure the model is properly trained with seasonal variations
    df['season'] = df['Date'].dt.month.map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    
    # Add seasonal interactions
    df['temp_season'] = df.apply(lambda x: x['Temperature at 2 meters (c)'] * (1 + 0.2 * np.sin(2 * np.pi * x['Month'] / 12)), axis=1)
    df['precip_season'] = df.apply(lambda x: x['Precipitation Corrected (mm/day)'] * (1 + 0.3 * np.cos(2 * np.pi * x['Month'] / 12)), axis=1)
    
    # Update the models dictionary with new features
    models['season'] = df['season']
    models['temp_season'] = df['temp_season']
    models['precip_season'] = df['precip_season']
    
    # Save the updated models
    with open('weather_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    with open('weather_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Models and scaler saved successfully!")

# Load models and scaler
def load_models():
    global models, scaler
    # Load trained models
    with open('weather_models.pkl', 'rb') as f:
        models = pickle.load(f)
    # Load scaler
    with open('weather_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Models and scaler loaded successfully!")

# Add after model training loop
save_models()  # Save models after training

# Update the main loop to include model loading option
while True:
    print("\n=== Weather Prediction System ===")
    print("1. Make a prediction by date")
    print("2. Load saved models")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            input_data = get_user_input()
            predictions = predict_weather(input_data)
            
            print("\nPredicted Weather Parameters:")
            print("-" * 50)
            for param, value in predictions.items():
                print(format_prediction(param, value))
        
        elif choice == '2':
            load_models()
            
        elif choice == '3':
            print("Thank you for using the Weather Prediction System!")
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.")
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try again.")
