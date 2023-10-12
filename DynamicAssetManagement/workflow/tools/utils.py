import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from statsmodels.tsa.stattools import adfuller
from tqdm.auto import tqdm

def read_file(file_name, path=None, index_col=None):
    """
    Read a CSV file and return it as a DataFrame.

    Parameters:
        file_name (str): The name of the CSV file without the extension.
        path (str, optional): The directory path where the file is located. Defaults to the 'data' directory.
        index_col (str or int, optional): Column to set as index of the DataFrame. None by default.

    Returns:
        DataFrame: The content of the CSV file as a DataFrame.
    """
    if not path:
        return pd.read_csv(f"workflow/data/{file_name}.csv", index_col=index_col, parse_dates=True)
    else:
        return pd.read_csv(f"workflow/{path}/{file_name}.csv", index_col=index_col, parse_dates=True)

def get_sectors():
    """
    Load and return the list of sectors from a pickled file.

    Returns:
        list: List of sectors.
    """
    with open("workflow/data/sector_list.pkl", "rb") as f:
        return pickle.load(f)
    
def get_snp():
    """
    Retrieve the S&P 500 ticker list from Wikipedia.

    Returns:
        list: List of tickers
    """
    snp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    snp_table = pd.read_html(snp_url)[0]
    return snp_table["Symbol"].str.replace(".", "-").to_list()

def set_plot_style():
    """
    Set the plot style for data visualization.

    This function sets the plot style to 'whitegrid' and configures line width and axis edge color.
    """
    sns.set_style("whitegrid")
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "k"

def test_stationarity(df):
    """
    Test the stationarity of each column in a DataFrame using the Augmented Dickey-Fuller (ADF) test.

    Parameters:
        df (pandas.DataFrame): DataFrame containing time series data, where each column represents a distinct time series.

    Returns:
        pandas.Series: A Series object containing the ADF test results for each column. 'Stationary' if p-value < 0.05, otherwise 'Non-Stationary'.
    """
    results = {}
    for col in tqdm(df.columns):
        adf_test = adfuller(df[col], autolag='AIC')
        p_value = adf_test[1]
        results[col] = "Stationary" if p_value < 0.05 else "Non-Stationary"

    return pd.Series(results)

