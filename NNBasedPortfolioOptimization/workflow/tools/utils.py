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