import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():

    """
    Sets the global plotting style using Seaborn and Matplotlib settings.
    
    This function customizes the plot to have a white grid background, 
    sets the line width for the plot, and defines the edge color of the axes.
    """

    sns.set_style("whitegrid")
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "k"

def plot_allocations(optimization, features, indices, dates, title):

    """
    Plots asset allocations and asset prices over a given date range.
    
    Parameters:
    - optimization (dict): A dictionary containing optimized weights for assets.
    - features (DataFrame): DataFrame containing asset features including prices.
    - indices (list): List of asset names.
    - dates (dict): Dictionary with 'start' and 'end' keys indicating the date range for plotting.
    - title (str): Title for the plot.
    
    The function creates a 2x2 subplot. Each subplot will display the allocation percentage
    for a specific asset alongside its price over the specified date range.
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=12)

    axes = axes.flatten()

    for ax, index in zip(axes, indices):

        ax.set_title(index)
        ax_twin = ax.twinx()

        index_allocation = optimization["WEIGHTS"][index]

        date_allocation = index_allocation.loc[dates["start"]:dates["end"]] * 100
        prices = features[indices].loc[dates["start"]:dates["end"]]

        date_allocation.plot(ax=ax)
        prices[index].plot(ax=ax_twin, grid=False, color="r", alpha=.5)

        ax.set_xlabel("")
        ax.legend(["Allocation"], loc='upper left', bbox_to_anchor=(0,1.2))
        ax_twin.legend(["Price"], loc='upper right', bbox_to_anchor=(1,1.2))

    fig.text(0.005, 0.5, 'Allocation (%)', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()