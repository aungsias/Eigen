![Eigen](eigen.png)

## About

Welcome to ***Eigen***, an evolving repository showcasing my work primarily in finance, logistics & operations, and machine learning (with occasional forays into domains). The repo serves as a conduit for my own intellectual curiosity, but I hope it engenders insights that are useful for everyone else. Regardless of whichever project you peruse, I hope you walk away having learned at least something!

Feel free to explore, and thank you for taking the time to visit my repository.

<sub>***Eigen was created to archive and distribute my ongoing endeavors is therefore subject to regular and frequent updates, both in content and structure.***</sub>

## Installation

To clone the repository and install required packages for this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/aungsias/Eigen.git

# Navigate to the project directory
cd Eigen

# Install required packages
pip install -r requirements.txt # Note: this may not work and you may have to manually install the dependencies.
```

## Contact Info

- Email: [aungsi.as99@gmail.com](mailto:aungsi.as99@gmail.com)
- LinkedIn: [Aung Si](https://www.linkedin.com/in/aungsi99)

## Project Catalogue

### Finance & the Capital Markets
- [`DynamicAssetManagement`](DynamicAssetManagement) - In this project, I engineered a dynamic, machine learning-driven framework for adaptive sector-based investment. Through biannual model recalibrations, the strategy remains agile, adjusting to market shifts and optimizing asset allocation. A 16-year backtest shows it outperforms traditional benchmarks like the S&P 500. The architecture incorporates risk management via a custom loss function, making it suitable for long-only portfolios. I also discuss its limitations and propose avenues for future refinement.

- [`DeepLearningPortfolioOptimization`](DeepLearningPortfolioOptimization) - This project focuses on using neural networks for optimizing investment portfolios, drawing inspiration from a [University of Oxford study](DeepLearningPortfolioOptimization/reference_paper/DeepLearningForPortfolioOptimization_Oxford.pdf) but with significant differences in implementation. This includes experimenting with both leveraged and non-leveraged investment strategies, opting out of volatility scaling, and using various activation functions in the models depdending on whether the strategy is leverage on or leverage off. The Long Short-Term Memory (LSTM) neural network, especially when applied to leveraged investments, demonstrated superior performance compared to other models and the VTI index across a ten-year backtesting period. This approach is also forward-looking, contrasting with the static nature of traditional Mean-Variance Optimization techniques. The LSTM models showed resilience and outperformance even within regimes of market turmoil, namely the Q1 market crash of 2020 caused by the COVID-19 pandemic.

### Operations & Logistics

- `Simulation` - TBD

### Miscellaneous

- `__tinkering__` - TBD

- `__operationsresearch__` - TBD