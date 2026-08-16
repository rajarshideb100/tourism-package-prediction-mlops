import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / 'best_tourism_package_model.joblib'
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title='Wellness Tourism Package Prediction', page_icon='✈️', layout='centered')
st.title('Wellness Tourism Package Prediction')
st.write('Predict whether a customer is likely to purchase the Wellness Tourism Package before contacting them.')

with st.form('customer_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=80, value=35)
        contact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
        city_tier = st.selectbox('City Tier', [1, 2, 3], index=0)
        duration = st.number_input('Duration of Pitch (minutes)', min_value=0.0, max_value=60.0, value=10.0)
        occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        persons = st.number_input('Number of Persons Visiting', min_value=1, max_value=10, value=3)
        followups = st.number_input('Number of Followups', min_value=0.0, max_value=10.0, value=3.0)
        product = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    with col2:
        stars = st.selectbox('Preferred Property Star', [3, 4, 5], index=0)
        marital = st.selectbox('Marital Status', ['Married', 'Divorced', 'Single'])
        trips = st.number_input('Number of Trips', min_value=0.0, max_value=25.0, value=3.0)
        passport = st.selectbox('Passport', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        satisfaction = st.selectbox('Pitch Satisfaction Score', [1, 2, 3, 4, 5], index=2)
        own_car = st.selectbox('Own Car', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        children = st.number_input('Number of Children Visiting', min_value=0.0, max_value=5.0, value=1.0)
        designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
        income = st.number_input('Monthly Income', min_value=0.0, max_value=100000.0, value=20000.0, step=500.0)

    submitted = st.form_submit_button('Predict Purchase')

if submitted:
    input_df = pd.DataFrame([{
        'Age': age, 'TypeofContact': contact, 'CityTier': city_tier,
        'DurationOfPitch': duration, 'Occupation': occupation, 'Gender': gender,
        'NumberOfPersonVisiting': persons, 'NumberOfFollowups': followups,
        'ProductPitched': product, 'PreferredPropertyStar': stars,
        'MaritalStatus': marital, 'NumberOfTrips': trips, 'Passport': passport,
        'PitchSatisfactionScore': satisfaction, 'OwnCar': own_car,
        'NumberOfChildrenVisiting': children, 'Designation': designation,
        'MonthlyIncome': income
    }])
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= 0.5)
    st.subheader('Prediction Result')
    if prediction == 1:
        st.success(f'Likely to purchase — probability: {probability:.1%}')
    else:
        st.info(f'Unlikely to purchase — probability: {probability:.1%}')
