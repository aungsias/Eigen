![LOGO](logo.png)

## About

Welcome to ***Eigen***, an evolving repository showcasing my work primarily in finance and machine learning, with occasional forays into other business and economics domains. Created to archive and disseminate my ongoing endeavors, Eigen is subject to regular and frequent updates, both in content and structure.

The projects within this repository are comprehensive undertakings aimed at solving intricate business conundrums, often necessitating an interdisciplinary approach that amalgamates various sectors and knowledge bases. These serve as conduits for intellectual curiosity, designed to engender insights that are useful at the very minimum. Regardless of whichever project you peruse, I hope you walk away having learned at least something!

Feel free to explore, and thank you for taking the time to visit my repository.

## Installation

To clone the repository and install required packages for this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/aungsias/Eigen.git

# Navigate to the project directory
cd Eigen

# Install required packages
pip install -r requirements.txt
```

## Contact Info

- Email: [aungsi.as99@gmail.com](mailto:aungsi.as99@gmail.com)
- LinkedIn: [Aung Si](https://www.linkedin.com/in/aungsi99)

## Project Catalogue

- [DynamicAssetManagement](DynamicAssetManagement) - In this project, I engineered a dynamic, machine learning-driven framework for adaptive sector-based investment. Through biannual model recalibrations, the strategy remains agile, adjusting to market shifts and optimizing asset allocation. A 16-year backtest shows it outperforms traditional benchmarks like the S&P 500. The architecture, comprehensive yet adaptable, incorporates risk management via a custom loss function, making it suitable for long-only portfolios. I also discuss its limitations and propose avenues for future refinement.

- [DeepLearningPortfolioOptimization](DeepLearningPortfolioOptimization) - The endeavor involves employing neural networks for portfolio optimization, inspired by a [paper from the University of Oxford](DeepLearningPortfolioOptimization/reference_paper/DeepLearningForPortfolioOptimization_Oxford.pdf) but diverges notably in execution. It explores both leveraged and non-leveraged strategies, with notable modifications like eschewing volatility scaling and employing different activation functions. The LSTM model, particularly in a leveraged setting, outperformed others and the VTI index significantly over a decade-long backtesting period. The methodology also showcases a forward-looking approach as opposed to the static nature of traditional Mean Variance Optimization. The detailed examination of models' performance during the COVID-19 downturn displayed the adaptability of LSTM models in asset allocation, substantiating the potential of neural networks in financial portfolio optimization.


