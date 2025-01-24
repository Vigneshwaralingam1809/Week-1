# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files

# Upload the dataset
uploaded = files.upload()

# Load the dataset
def load_data(file_path):
    """Loads the dataset from the given file path."""
    data = pd.read_csv(file_path)
    return data

# Preprocess data and split into features and target
def preprocess_data(data):
    """Splits data into features (X) and target (y)."""
    X = data.drop(columns=['generated_power_kw'])  # Features
    y = data['generated_power_kw']  # Target variable
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
def train_model(X_train, y_train):
    """Trains a Linear Regression model and returns it."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns MSE and R2 Score."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# Visualize the predictions
def visualize_results(y_test, y_pred):
    """Visualizes actual vs predicted solar power output."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             color='red', linestyle='--', label='Ideal Fit')
    plt.title('Actual vs Predicted Solar Power Output')
    plt.xlabel('Actual Power Output (kW)')
    plt.ylabel('Predicted Power Output (kW)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
if __name__ == "__main__":
    # File path to the uploaded dataset
    file_path = "03f4d1c1a55947025601 (1).csv"  # Automatically get uploaded file name
    
    # Step 1: Load the data
    data = load_data(file_path)
    
    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Step 3: Train the model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Step 5: Visualize the results
    visualize_results(y_test, y_pred)
