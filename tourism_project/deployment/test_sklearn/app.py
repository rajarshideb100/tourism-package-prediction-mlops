import streamlit as st
import pandas as pd
import sklearn

st.title("scikit-learn Deployment Test")
st.write("Streamlit started successfully.")
st.write("Pandas version:", pd.__version__)
st.write("scikit-learn version:", sklearn.__version__)
