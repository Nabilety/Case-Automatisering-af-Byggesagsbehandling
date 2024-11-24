"""
Modul: Automatisk klassifikation af byggeansøgninger
Formål: At klassificere byggeansøgninger baseret på typen af byggeprojekt.

Dato: 23-11-2024
Version: 1.0

Afhængigheder:
- pandas: Databehandling
- scikit-learn: Klassifikationsalgoritme
- joblib: Modellagring

Eksempel på brug:
1. Indlæs byggeansøgningsdata fra en CSV-fil.
2. Klassificer hver ansøgning baseret på de tilgængelige data.
3. Eksporter resultaterne til en ny fil for videre behandling.

NB: Løsning afhænger af data
-> formular (budget, lokation mm.)
-> lang brødtekst
"""


# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
# Assume 'building_applications.csv' was generated and contains target column 'Project_Type'
# Extracted from a dataformular
data = pd.read_csv('datasæt.csv')
data = pd.DataFrame(data)

# Data Preprocessing
print("Step 1: Preprocessing data...")
# Map the target column 'Project_Type' to numeric values for classification
data['projekt_type'] = data['projekt_type'].map({
    "Tilbygning": 0,
    "Nybyg": 1,
    "Renovering": 2
})

data.dropna(subset=['projekt_type'], inplace=True)  # Remove rows with missing values
data.reset_index()

X = data.drop(columns=["projekt_type"])  # Features: application descriptions
y = data["projekt_type"] # Labels: project types

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training
print("Step 3: Training the classifier...")
model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=3)
model.fit(X_train, y_train)

# Model Evaluation
print("Step 4: Evaluating the model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()

# Confusion Matrix - for the 20 test data rows
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Tilbygning", "Nybyg", "Renovering"])
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Visualization
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature')
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()



# Save model for reuse
joblib.dump(model, 'random_forest_building_classifier.pkl')

print("Model and features saved successfully!")

# Usage Example (Deployment Simulation)
def classify_building_application(input_data):
    """
    Classify a building application based on input features.

    Parameters:
        input_data (dict): A dictionary containing the feature values for the application.

    Returns:
        str: Predicted project type (e.g., 'Tilbygning', 'Nybyg', 'Renovering').
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the input DataFrame has the same feature columns as the training data
    input_df = pd.get_dummies(input_df, drop_first=True).reindex(columns=X.columns, fill_value=0)

    # Load the trained model
    model = joblib.load('random_forest_building_classifier.pkl')

    # Predict project type
    prediction = model.predict(input_df)[0]
    return prediction


# Example usage of the classify_building_application function
example_input = {
    "Building_Area": 250,
    "Project_Duration": 12,
    "Budget": 150000
}

predicted_type = classify_building_application(example_input)
print(f"Predicted Project Type: {predicted_type}")

