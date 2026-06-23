import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

st.title("Tesla Stock Price Prediction App")
st.write("This app predicts the behavior of Tesla's closing price for 1, 5, and 10 days ahead using Deep Learning.")

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

@st.cache_resource
def load_data():
    df = pd.read_csv('TSLA.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.ffill(inplace=True)
    return df

@st.cache_resource
def load_keras_model(model_name, horizon):
    filename = f"{model_name}_{horizon}d.keras"
    return tf.keras.models.load_model(filename)

scaler = load_scaler()
df = load_data()

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "SimpleRNN"])
horizon_choice = st.sidebar.selectbox("Prediction Horizon (Days)", [1, 5, 10])

st.subheader("Historical Stock Price")
st.line_chart(df['Adj Close'])

st.subheader(f"Predicting Next {horizon_choice} Days using {model_choice}")

# Prepare the last 60 days of data
last_60_days = df['Adj Close'].values[-60:]
scaled_last_60 = scaler.transform(last_60_days.reshape(-1, 1))
X_input = np.reshape(scaled_last_60, (1, 60, 1))

model = load_keras_model(model_choice, horizon_choice)
if st.button("Predict"):
    prediction = model.predict(X_input)
    prediction_inv = scaler.inverse_transform(prediction)[0]
    
    st.write(f"### Predicted Prices for the next {horizon_choice} day(s):")
    
    # Generate future dates (skip weekends)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_choice)
    
    pred_df = pd.DataFrame({"Predicted Adj Close": prediction_inv}, index=future_dates)
    st.dataframe(pred_df)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index[-60:], df['Adj Close'].values[-60:], label="Past 60 Days", color='blue')
    ax.plot(future_dates, prediction_inv, label="Predicted", color='red', marker='o')
    ax.set_title(f"Tesla Stock Price Forecast ({model_choice})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
