# app.py
import pandas as pd
import streamlit as st

# Load CSV
csv_file = "data/stream_output/predictions/predictions.csv"
df = pd.read_csv(csv_file)

# Show only first 10 columns
df_display = df.iloc[:, :10]

st.title("Predictions Table")
st.write("Showing first 10 columns of predictions.csv")
st.dataframe(df_display)  # interactive table
