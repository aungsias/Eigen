import streamlit as st
import datetime as dt
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from math import sqrt

# Load the CSV
prices = pd.read_csv("workflow/data/snp_prices.csv", index_col=0, parse_dates=True)
available_tickers = prices.columns.to_list()

page = st.sidebar.selectbox(
    'Choose a page',
    ('View Stocks', 'Our Services')
)

if page == 'View Stocks':
    st.title("Stock Price Dashboard")

    # User Input
    ticker = st.selectbox("Choose Stock Ticker:", available_tickers, index=0)

    # Date range options
    date_option = st.selectbox('Select Time Range:', ['Custom', 'YTD', '1Y', '5Y'])
    
    if date_option == 'Custom':
        start_date = st.date_input("Start Date:", value=dt.datetime(2015, 1, 1))
        end_date = st.date_input("End Date:", value=dt.datetime.now())
    elif date_option == 'YTD':
        start_date = dt.datetime(dt.datetime.now().year, 1, 1)
        end_date = dt.datetime.now()
    elif date_option == '1Y':
        start_date = dt.datetime.now() - dt.timedelta(days=365)
        end_date = dt.datetime.now()
    elif date_option == '5Y':
        start_date = dt.datetime.now() - dt.timedelta(days=1825)
        end_date = dt.datetime.now()

    # Fetch Data
    stock_data = prices.loc[(prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date)), [ticker]]

    if stock_data.empty:
        st.write("No data available.")
    else:
        # Volatility Calculation
        annual_volatility = stock_data[ticker].pct_change().std() * sqrt(252)
        st.write(f"Annual Volatility: {annual_volatility:.2%}")

        # Table Display
        show_table = st.checkbox("See Table")
        if show_table:
            st.dataframe(stock_data)

        # Download CSV
        csv_buffer = BytesIO()
        stock_data.to_csv(csv_buffer)
        csv_buffer.seek(0)
        st.download_button(
            "Download CSV",
            csv_buffer,
            file_name=f"{ticker}_{start_date}_{end_date}.csv",
            mime="text/csv"
        )

        # Plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data[ticker],
                mode='lines',
                name='Closing Price',
                line=dict(color='green')
            )
        )

        fig.update_layout(
            title=f"{ticker} Closing Prices",
            xaxis_title="Date",
            yaxis_title="Closing Price (USD)",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

elif page == 'Our Services':
    st.title("Our Services")
    st.write("Information about our services.")
