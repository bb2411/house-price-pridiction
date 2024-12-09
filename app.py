import pickle
import pandas as pd
import streamlit as st
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
original_data = pd.read_csv('data.csv')
original_data['statezip'] = original_data['statezip'].str.replace('WA ', '')
unique_zip_codes = original_data['statezip'].unique()
feature_columns = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", 
    "waterfront", "view", "condition", "sqft_basement", "yr_built", "yr_renovated", "statezip"
]
def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df['statezip'] = pd.to_numeric(input_df['statezip'])
    st.write("Input Data for Prediction:")
    st.write(input_df)
    prediction = model.predict(input_df)
    return prediction[0]
def main():
    st.title("House Price Prediction")
    st.write("Enter the property details to predict the price:")
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
    sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
    sqft_lot = st.number_input("Lot Area (sqft)", min_value=500, max_value=50000, value=5000)
    floors = st.number_input("Number of Floors", min_value=0, max_value=5, value=1)
    waterfront = st.selectbox("Waterfront View", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    view = st.number_input("View Rating", min_value=0, max_value=4, value=0)
    condition = st.number_input("Condition Rating", min_value=1, max_value=5, value=3)
    sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=0)
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=1990)
    yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2024, value=0)
    selected_zip = st.selectbox("Zip Code", unique_zip_codes)
    input_data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "statezip": int(selected_zip) 
    }
    if st.button("Predict Price"):
        try:
            price = predict_price(input_data)
            st.success(f"The predicted price of the house is ${price:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    st.markdown("---")
    st.markdown("**Author:** Darshan Jadav & Bhargav Bhatt")
if __name__ == "__main__":
    main()
