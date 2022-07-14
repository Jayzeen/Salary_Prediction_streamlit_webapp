import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)

    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# Prediction function
def show_predict_page():
    st.title("Software Developer Salary Prediction")
     
    st.write("""### We need Data to Predict the Salary""")
     

    # Which Country needed as input
    countries = (
        "United States of America",
        "India",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
        "Turkey",
        "Switzerland",
        "Israel",
        "Norway",
    )

    # Education levels needed as input
    education = (
        
    'Master’s degree',
    'Bachelor’s degree',
    'Post grad',
    'Less than a Bachelors',
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    # Submit button
    submitted = st.button("Calculate Salary")

    if submitted:
        x = np.array([[ country, education, experience ]])
        
        # Encode the data
        x[:, 0] = le_country.transform(x[:, 0])
        x[:, 1] = le_education.transform(x[:, 1])
        x = x.astype(float)
        
        salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
