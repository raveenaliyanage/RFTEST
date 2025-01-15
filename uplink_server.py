import anvil.server
import joblib
import pandas as pd

# Connect to Anvil app
anvil.server.connect("server_6QTKQ5XCFOPXH4TODFEO4JKU-3X5E4NTVNHIEELKH")  # Replace with your actual Uplink key

# Load your model and columns
model = joblib.load('forest_model.pkl')  # Ensure this file is in the same folder
model_columns = joblib.load('model_columns.pkl')

@anvil.server.callable
def predict_price(user_input):
    """
    Predict the highest and average price based on user input.
    """
    print(f"Received input data: {user_input}")
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical variables (like 'Grade' and 'Location')
    input_encoded = pd.get_dummies(input_df, columns=['Grade', 'Location'], drop_first=False)

    # Reindex to match the model's expected column order
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    print(f"Reindexed input data: {input_encoded}")

    # Make predictions
    predictions = model.predict(input_encoded)
    return {
        'Highest_Price': predictions[0][0],  # First column in predictions
        'Average_Price': predictions[0][1]   # Second column in predictions
    }

# Keep the server running to listen for Anvil calls
anvil.server.wait_forever()
