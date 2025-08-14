import streamlit as st
import numpy as np
import joblib

# Load the model
kmeans = joblib.load(r"C:\Users\Mega Store\Videos\ML\last\kmeans_model.pkl")

st.title("Customer Segmentation (KMeans)")

st.write("Enter the following values to find your cluster:")

# User inputs
fresh = st.number_input("Fresh", min_value=0, step=1)
milk = st.number_input("Milk", min_value=0, step=1)
grocery = st.number_input("Grocery", min_value=0, step=1)
frozen = st.number_input("Frozen", min_value=0, step=1)
detergents_paper = st.number_input("Detergents_Paper", min_value=0, step=1)
delicassen = st.number_input("Delicassen", min_value=0, step=1)

if st.button("Predict Cluster"):
    # Prepare data for prediction
    input_data = np.array([[fresh, milk, grocery, frozen, detergents_paper, delicassen]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"You belong to cluster number: {cluster}")