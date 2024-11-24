# Case-Automatisering-af-Byggesagsbehandling
![GitHub last commit](https://img.shields.io/github/last-commit/nabilety/Case-Automatisering-af-Byggesagsbehandling)
# Automating Building Application Screening with AI

1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Features](#features)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Model details](#model-details)
7. [Performance metrics](#performance-metrics)
8. [Deployment](#deployment)
9. [Contribution](#contribution)

## Overview
Rødovre Kommune aims to improve efficiency in building application processing by automating the initial screening phase. This project utilize machine learning to classify building applications based on project type (e.g., new construction, renovation, addition). The solution is designed for scalability and compliance with relevant AI regulations.

## Project Goals
- **Efficiency**: Automate classification to reduce manual workload.
- **Accuracy**: Ensure reliable and transparent predictions.
- **Scalability**: Design a reusable pipeline for future applications.

## Features
- **Text Classification**: Uses a machine learning model to classify building descriptions into predefined types.
- **Feature Extraction**: Extract features from mathematical tools such as Feature Importance and Correlation Matrix.
- **Deployment Ready**: Provides saved models for integration with Rødovre Kommune's IT systems.

---

## Installation and Setup

### Install Python on Windows/MacOS
Download and use the python installer, if you are shown the option “Disable path length limit” it is
recommended to activate that option to prevent possible future problems with long paths. Installers
for the most up to date versions are found [here](https://www.python.org/downloads/). Select the newest release and download the right installer according to your operating system.

### Install Python on Linux:
Execute these commands in your terminal:
   ```console
   sudo apt-get update -y
   sudo apt-get install -y python3
   ```
### Clone Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/building-application-ai.git
   cd building-application-ai
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```
3. Ensure your dataset is in the same directory, formatted as a CSV with two columns:
   * <mark>description:</mark> Text description of the building application.
   * <mark>type:</mark> Type of project (e.g., renovation, new construction, extension).

## Create virtual environment
Create and activate virtual environment
  ```bash
   python -m venv .venv
   # On Windows (PowerShell or Command Prompt):
   .\.venv\Scripts\activate
   # On Windows (Git Bash):
   source .venv/Scripts/activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```
---
## Usage
1. Training and Evaluation
   Run the byggeansøgning_klassifikation.py script on your terminal to train the model and evaluate its performance:
   ```console
   python byggeansøgning_klassifikation.py
   ```
2. Classification Example
   After training, use the saved model to classify new applications:
   ```python
   from byggeansøgning_klassifikation import classify_application

   example_description = "Adding a new floor to the building." # Example of extension
   predicted_type = classify_application(example_description) # Predict
   print("Predicted Type:", predicted_type) # Answer
   ```
3. Visualizations
   The script includes visualizations for:
   * Data Distribution: Bar chart of application types.
     - Description: Bar chart displaying the frequency of each application type.
     - Purpose: Identify data imbalances that might affect model training.
   * Feature Importance: Top 10 features identified by the model.
     - Description: Horizontal bar chart of the top 10 features contributing to model decisions.
     - Purpose: Explain which terms are most significant for classification.
   * Confusion Matrix: Performance breakdown by type.
     - Description: Heatmap of model performance across all classes.
     - Purpose: Diagnose misclassifications and understand the model's weaknesses.

---
## Model Details
* Algorithm: Random Forest Classifier
* Feature Extraction: Feature Importance and Correlation Matrix
* Evaluation: Tested on a split dataset (80% training, 20% testing).

### Performance Metrics
* Accuracy: Typically exceeds 85% on well-labeled datasets.
* Classification Report: Includes precision, recall, and F1-score

---
## Deployment
The model and features are saved as <mark>.pkl</mark> files and can be integrated into any Python-based system.

---
## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. PLease make sure to update tests as appropriate
