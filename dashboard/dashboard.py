# Import libraries
import streamlit as st
import datetime
import plotly.graph_objects as go
from dashboard_helper import *

# Title for data fetching section
st.title("Stock Price Dashboard")

# Default start date
default_start_date = datetime.date(2010, 1, 1)

# Date range picker for historical data
start_date = st.date_input("Start Date", value=default_start_date)
end_date = st.date_input("End Date")

# Fetch button to initiate data fetching
fetch_button = st.button("Fetch S&P 500 Data")

# Initialize an empty DataFrame or get cached data
data = None

# Fetch and cache data
if fetch_button:
    data = fetch_data(start_date, end_date)
    st.session_state.data = data  # Store data in session state

# Check if data exists in session state
if 'data' in st.session_state:
    data = st.session_state.data

    # List of stock tickers
    tickers = data["Adj Close"].columns.tolist()

    # User selection of stock
    selected_stock = st.selectbox('Select a stock:', tickers)

    # Create a Plotly figure
    fig = go.Figure()

    # Add only the adjusted closing prices, with green color
    fig.add_trace(go.Scatter(x=data.index,
                             y=data['Adj Close'][selected_stock],
                             mode='lines',
                             name='Adj Close',
                             line=dict(color='green')))

    # Add volume as a bar chart (oscillator)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'][selected_stock], name='Volume', yaxis='y2'))

    # Update layout for dual y-axis and fix x-axis positioning
    fig.update_layout(yaxis=dict(domain=[0.3, 1]),
                      yaxis2=dict(domain=[0, 0.2]),
                      xaxis=dict(anchor="y"),
                      title=f"{selected_stock} Price and Volume",
                      yaxis_title="Adj Close",
                      yaxis2_title="Volume",
                      template='plotly_dark')

    # Show figure in Streamlit
    st.plotly_chart(fig)
