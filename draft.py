import streamlit as st
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

# Title
st.title("Stock Price Dashboard")

# User input
ticker = st.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.date_input("Start Date:", value=dt.datetime(2015, 1, 1))
end_date = st.date_input("End Date:", value=dt.datetime.now())

# Fetch stock data
stock_data = yf.download(ticker, start=start_date, end=end_date)

if stock_data.empty:
    st.write("No data available for the given dates.")
else:
    # Option to display table
    show_table = st.checkbox("See Table")
    if show_table:
        st.dataframe(stock_data)
    
    # Download CSV button
    csv_buffer = BytesIO()
    stock_data.to_csv(csv_buffer)
    csv_buffer.seek(0)
    st.download_button(
        label="Download CSV",
        data=csv_buffer,
        file_name=f"{ticker}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

    # Plotting with Plotly
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name='Closing Price',
            line=dict(color='green')
        )
    )

    # Customize layout
    fig.update_layout(
        title=f"{ticker} Closing Prices",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        template="plotly_dark",
        xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2),
        yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='rgb(204, 204, 204)'),
    )

    st.plotly_chart(fig)
