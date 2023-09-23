import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
        return pd.read_csv(f"data/{file_name}.csv", index_col=index_col, parse_dates=True)
    else:
        return pd.read_csv(f"{path}/{file_name}.csv", index_col=index_col, parse_dates=True)
    
def save_models(models: dict):
    """
    Serialize and save a dictionary of models to a pickled file.

    Parameters:
        models (dict): Dictionary containing model names as keys and trained model objects as values.

    The function stores the models in a pickled file located in the 'data' directory, with the file name 'models.pkl'.
    """
    with open("data/models.pkl", "wb") as f:
        pickle.dump(models, f)
        
def load_models():
    """
    Deserialize and load a dictionary of models from a pickled file.

    Returns:
        dict: Dictionary containing model names as keys and trained model objects as values.

    The function reads the models from a pickled file located in the 'data' directory, with the file name 'models.pkl'.
    """
    with open("data/models.pkl", "rb") as f:
        models = pickle.load(f)
    return models


def get_sectors():
    """
    Load and return the list of sectors from a pickled file.

    Returns:
        list: List of sectors.
    """
    with open("data/sector_list.pkl", "rb") as f:
        return pickle.load(f)

def set_plot_style():
    """
    Set the plot style for data visualization.

    This function sets the plot style to 'whitegrid' and configures line width and axis edge color.
    """
    sns.set_style("whitegrid")
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "k"
