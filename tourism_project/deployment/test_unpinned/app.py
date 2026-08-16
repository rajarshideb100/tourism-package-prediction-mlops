import streamlit as st
import pandas as pd
import sklearn
import xgboost
import joblib

st.title("Unpinned ML Stack Deployment Test")
st.write("Streamlit started successfully.")
st.write("Pandas:", pd.__version__)
st.write("scikit-learn:", sklearn.__version__)
st.write("XGBoost:", xgboost.__version__)
st.write("joblib:", joblib.__version__)
