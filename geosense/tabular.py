import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'C:\Users\User\imageclassy\ashok\dataset\tabulardata\synthetic_soil_bearing_capacity.csv')

print("="*80)
print("SOIL BEARING CAPACITY PREDICTION MODEL")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Separate features and targets
X = df[['soil_type', 'moisture_content', 'bulk_density', 'grain_size', 
        'foundation_depth', 'field_observation_score']]
y_bearing = df['bearing_capacity']
y_safety = df['safety_class']

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Encode categorical variables
le_soil = LabelEncoder()
le_grain = LabelEncoder()
le_safety = LabelEncoder()

X['soil_type_encoded'] = le_soil.fit_transform(X['soil_type'])
X['grain_size_encoded'] = le_grain.fit_transform(X['grain_size'])
y_safety_encoded = le_safety.fit_transform(y_safety)

print("\nSoil Type Encoding:")
for i, label in enumerate(le_soil.classes_):
    print(f"  {label}: {i}")

print("\nGrain Size Encoding:")
for i, label in enumerate(le_grain.classes_):
    print(f"  {label}: {i}")

print("\nSafety Class Encoding:")
for i, label in enumerate(le_safety.classes_):
    print(f"  {label}: {i}")

# Select encoded features
X_final = X[['soil_type_encoded', 'moisture_content', 'bulk_density', 
             'grain_size_encoded', 'foundation_depth', 'field_observation_score']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

print(f"\nFeature Matrix Shape: {X_scaled.shape}")
print(f"Target 1 (Bearing Capacity) Shape: {y_bearing.shape}")
print(f"Target 2 (Safety Class) Shape: {y_safety_encoded.shape}")

# Split data for both targets
X_train, X_test, y_bearing_train, y_bearing_test = train_test_split(
    X_scaled, y_bearing, test_size=0.2, random_state=42
)

_, _, y_safety_train, y_safety_test = train_test_split(
    X_scaled, y_safety_encoded, test_size=0.2, random_state=42
)

print(f"\nTraining Set Size: {X_train.shape[0]}")
print(f"Test Set Size: {X_test.shape[0]}")

print("\n" + "="*80)
print("MODEL 1: BEARING CAPACITY PREDICTION (REGRESSION)")
print("="*80)

# Train Random Forest Regressor for bearing capacity
rf_regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest Regressor...")
rf_regressor.fit(X_train, y_bearing_train)
print("Training completed!")

# Predictions
y_bearing_pred_train = rf_regressor.predict(X_train)
y_bearing_pred_test = rf_regressor.predict(X_test)

# Evaluation metrics for bearing capacity
print("\n--- BEARING CAPACITY MODEL EVALUATION ---")
print("\nTraining Set Performance:")
print(f"  R² Score: {r2_score(y_bearing_train, y_bearing_pred_train):.4f}")
print(f"  MAE: {mean_absolute_error(y_bearing_train, y_bearing_pred_train):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_bearing_train, y_bearing_pred_train)):.4f}")

print("\nTest Set Performance:")
print(f"  R² Score: {r2_score(y_bearing_test, y_bearing_pred_test):.4f}")
print(f"  MAE: {mean_absolute_error(y_bearing_test, y_bearing_pred_test):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_bearing_test, y_bearing_pred_test)):.4f}")

# Feature importance
feature_names = ['Soil Type', 'Moisture Content', 'Bulk Density', 
                'Grain Size', 'Foundation Depth', 'Field Observation Score']
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_regressor.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Bearing Capacity):")
for idx, row in importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "="*80)
print("MODEL 2: SAFETY CLASS PREDICTION (CLASSIFICATION)")
print("="*80)

# Train Random Forest Classifier for safety class
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest Classifier...")
rf_classifier.fit(X_train, y_safety_train)
print("Training completed!")

# Predictions
y_safety_pred_train = rf_classifier.predict(X_train)
y_safety_pred_test = rf_classifier.predict(X_test)

# Evaluation metrics for safety class
print("\n--- SAFETY CLASS MODEL EVALUATION ---")
print("\nTraining Set Performance:")
print(f"  Accuracy: {accuracy_score(y_safety_train, y_safety_pred_train):.4f}")

print("\nTest Set Performance:")
print(f"  Accuracy: {accuracy_score(y_safety_test, y_safety_pred_test):.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_safety_test, y_safety_pred_test, 
                          target_names=le_safety.classes_))

print("Confusion Matrix (Test Set):")
cm = confusion_matrix(y_safety_test, y_safety_pred_test)
print(cm)

# Feature importance for classification
importance_df_clf = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Safety Class):")
for idx, row in importance_df_clf.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "="*80)
print("SAVING TRAINED MODELS")
print("="*80)

# Save models and preprocessors
models = {
    'bearing_capacity_model': rf_regressor,
    'safety_class_model': rf_classifier,
    'scaler': scaler,
    'soil_type_encoder': le_soil,
    'grain_size_encoder': le_grain,
    'safety_class_encoder': le_safety,
    'feature_names': feature_names
}

with open('soil_bearing_model.pkl', 'wb') as f:
    pickle.dump(models, f)

print("\n✓ Models saved to 'soil_bearing_model.pkl'")

# Save predictions to CSV
results_df = pd.DataFrame({
    'Actual_Bearing_Capacity': y_bearing_test.values,
    'Predicted_Bearing_Capacity': y_bearing_pred_test,
    'Bearing_Error': y_bearing_test.values - y_bearing_pred_test,
    'Actual_Safety_Class': le_safety.inverse_transform(y_safety_test),
    'Predicted_Safety_Class': le_safety.inverse_transform(y_safety_pred_test),
})

results_df.to_csv('model_predictions.csv', index=False)
print("✓ Predictions saved to 'model_predictions.csv'")

print("\n" + "="*80)
print("PREDICTION EXAMPLE")
print("="*80)

# Example prediction function
def predict_new_sample(soil_type, moisture_content, bulk_density, 
                      grain_size, foundation_depth, field_observation_score):
    """
    Predict bearing capacity and safety class for new soil sample
    """
    # Encode categorical features
    soil_encoded = le_soil.transform([soil_type])[0]
    grain_encoded = le_grain.transform([grain_size])[0]
    
    # Create feature array
    features = np.array([[soil_encoded, moisture_content, bulk_density,
                         grain_encoded, foundation_depth, field_observation_score]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    bearing_pred = rf_regressor.predict(features_scaled)[0]
    safety_pred_encoded = rf_classifier.predict(features_scaled)[0]
    safety_pred = le_safety.inverse_transform([safety_pred_encoded])[0]
    
    # Get prediction probabilities
    safety_proba = rf_classifier.predict_proba(features_scaled)[0]
    
    return bearing_pred, safety_pred, safety_proba

# Test with example
print("\nExample Prediction:")
print("Input: Soil Type=Sand, Moisture=28.35%, Bulk Density=16.97 kN/m³")
print("       Grain Size=Fine, Foundation Depth=1.72m, Field Score=3")

bearing, safety, proba = predict_new_sample(
    'Sand', 28.35, 16.97, 'Fine', 1.72, 3
)

print(f"\nPredicted Bearing Capacity: {bearing:.2f} kN/m²")
print(f"Predicted Safety Class: {safety}")
print("\nClass Probabilities:")
for i, class_name in enumerate(le_safety.classes_):
    print(f"  {class_name}: {proba[i]:.4f} ({proba[i]*100:.2f}%)")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nFiles Generated:")
print("  1. soil_bearing_model.pkl - Trained models and preprocessors")
print("  2. model_predictions.csv - Test set predictions and errors")
print("\nTo use the model for predictions, load the pickle file and use")
print("the predict_new_sample() function with your input parameters.")
print("="*80)