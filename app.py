import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


st.title("📈 Stock Price Forecast App (SARIMA)")


@st.cache_data
def load_data():
    df = pd.read_csv("your_stock_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


df = load_data()


st.subheader("Recent Stock Prices")
st.line_chart(df["Close"])


days = st.slider("Forecast Days", 1, 60, 30)


if st.button("Predict"):
    st.write("⏳ Training SARIMA model...")


    model = SARIMAX(df["Close"], order=(1,1,1), seasonal_order=(1,1,1,252))
    result = model.fit(disp=False)


    forecast = result.get_forecast(steps=days)
    predicted = forecast.predicted_mean
    conf = forecast.conf_int()


    st.subheader("Predicted Values")
    st.write(predicted)


    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Close"], label="History")
    ax.plot(predicted, label="Forecast", linestyle='--')
    ax.fill_between(conf.index, conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
    ax.legend()



    st.pyplot(fig)
