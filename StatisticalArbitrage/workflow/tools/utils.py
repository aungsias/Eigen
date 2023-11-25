import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "k"