# Portfolio Optimization via Neural Networks

Aung Si<br>
October 21<sup>st</sup>, 2023

---

## Contents
- [1. Overview](#overview)
- [2. The Staticity of Mean Variance Optimization](#the-staticity-of-mean-variance-optimization)
- [3. Methodology & Data](#methodology--data)
    - [3.1. Feature Engineering](#31-feature-engineering)
    - [3.2. Sidestepping Price / Return Forecasting](#32-sidestepping-price--return-forecasting)
    - [3.3. Data Sources](#33-data-sources)
    - [3.4. Temporal Considerations](#34-temporal-considerations)
- [4. Modeling](#4-modeling)
7. Results
8. Conclusion
9. Limitations
10. Future Work
11. Repository Structure

## 1. Overview

*This endeavor draws substantive influence from the University of Oxford's [Deep Learning for Portfolio Optimization](https://arxiv.org/pdf/2005.13665.pdf). However, the project diverges at crucial junctures in its execution.*

The original paper, written by Zihao Zhang, Stefan Zohren, and Stephen Roberts, introduces a novel approach to portfolio optimization by employing neural networks. Unlike traditional mean-variance methods, which are inherently static and rely on historical data, the paper's approach uses the Sharpe ratio as the objective function (maximized via gradient ascent) to adaptively optimize portfolio allocations. This creates a more dynamic, forward-looking model capable of adjusting to market conditions. While the authors tested various models such as FCN, CNN, and LSTM using a long-only strategy without leverage, my exploration included both with and without leverage approaches. The critical distinction lies in the application of sigmoid and softmax functions.

Another noteworthy divergence is in volatility scaling: while the authors applied it, I opted against its implementation for the sake of concision. It should be noted here that volatility scaling inherently employs leverage, so in some sense, the weights output by the sigmoid model that I employed share some similarities with the authors' volatility-scaled allocations. Despite the differences, I corroborated their conclusion that the LSTM model exhibited the most adaptability and consistently superior performance. Specifically, in my rendition, the leveraged LSTM model surpassed the VTI index by 152.4%, boasting an impressive Sharpe ratio of 1.12 versus VTI's 0.59 over a decade-long backtesting period. Conversely, the non-leveraged LSTM model yielded a 50.9% advantage over the VTI and recorded a Sharpe ratio of 1.09 against VTI's 0.59.

A salient characteristic of the sigmoid function, especially pertinent to my leveraged model, is its behavior. On certain dates, it results in leveraging, while on others, not all available capital is deployed. This variability speaks to the model's adaptability to market dynamics, optimizing capital utilization based on prevailing conditions.

Lastly, while the original paper furnished an architecture template, the exact model specifications were not touched upon. The task of defining the models, therefore, was undertaken independently, relying on the provided blueprint yet necessitating intricate detailing on my part.

All implementations of neural networks in this project were carried out with [`PyTorch`](https://pytorch.org).

## 2. The Staticity of Mean Variance Optimization

Traditional portfolio optimization has long been dominated by the [Mean Variance Optimization (MVO)](https://www.math.hkust.edu.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf) method, introduced by Harry Markowitz in 1952. While this method is grounded in elegant mathematics and provides a pragmatic foundation for portfolio selection, it exhibits notable shortcomings.

MVO hinges on the assumption that asset returns are normally distributed, and it primarily focuses on two parameters: expected returns and the covariance matrix of returns. By optimizing these, MVO determines the best possible asset allocation that will yield the highest expected return for a given level of risk.

However, the static nature of MVO emerges from its reliance on historical data to gauge future asset performances. By design, it takes a retrospective stance, relying on past return distributions to craft future portfolio allocations. This backward-looking approach intrinsically limits its adaptability to evolving market conditions and new information sets.

Moreover, the assumptions of MVO become especially questionable during tumultuous market periods. Black swan events, such as financial crises, can skew return distributions away from normality, leading to estimation errors in MVO's output. Additionally, the challenge of accurately estimating expected returns amplifies the potential pitfalls of the MVO. Often, slight changes in expected return estimations can result in substantially different portfolio compositions, creating inconsistency and unpredictability. MVO is also deterministic in its nature, which can lead to extreme portfolio weights. Without constraints, it might recommend full allocation to a singular asset if historical data suggests superior returns, disregarding diversification principles.

In light of these limitations, the quest for a more dynamic, adaptive, and forward-looking portfolio optimization technique becomes paramount. As the financial industry evolves, the inclusion of machine learning and neural networks offers a promising avenue to address the staticity inherent in MVO, laying the groundwork for the methodologies explored in subsequent sections of this study.

## 3. Methodology & Data

The methodology underpinning this study aligns closely with the framework presented in the original paper. Both our approaches center on the use of asset prices and returns as input features. I follow the below structure.

1. Retrieve market index data (VTI, DBC, AGG, and VIX) and compute logarithmic daily returns.
2. Conduct walk-forward portfolio optimization via neural networks (leveraged and non-leveraged).
3. Conduct a backtest for both scenarios, ***accounting for both transaction costs, borrowing costs, and principal ammortization***.
5. Examine allocations at specific timeframes (namely the COVID-19 pandemic) and overall adaptability of allocations with respect to the price trajectories of each index.
6. Examine the usage of leverage over time of each model.

While I maintain the core principles of the original paper, the key divergenceS lie in the dates I use for the data and in my usage of the sigmoid function, and which I will explore later.

### 3.1 Feature Engineering
The sole features employed are the asset prices and returns. This parsimony in feature selection helps mitigate the risk of overfitting, which may help to ensure the models remain robust and generalize well across different market conditions. However, the authors note the inherent potential for feature redundancy, given that returns are inherently derived from prices. Their inclusion hinges on their role in computing overall backtest returns. 

### 3.2 Sidestepping Price / Return Forecasting
While traditional methods often require a forecast of future returns or prices, this approach obviates that need. Given the inherent unpredictability, near-randomness, and self-correcting nature of the stock market, attempting to forecast returns frequently results in limited practical insights and can introduce unnecessary noise (though it does work in some cases, as evident in a [previous project I've worked on](https://github.com/aungsias/Eigen/tree/main/DynamicAssetManagement).)

Instead, the innovation brought forth by the authors, and mirrored in this study, is the direct translation of this information into optimal portfolio weights. This is achieved using neural networks, specifically through the activation functions which are central to our model differentiation: **sigmoid** (where outputs range from 0 to 1) for leveraged models and **softmax** (where outputs range from 0 to 1 sum up to 1) for non-leveraged ones.

### 3.3 Data Sources
All data for this project were retrieved via [`yfinance`](https://pypi.org/project/yfinance/). Both the original study and this research use the same four indices as primary data sources:

- **VTI (Vanguard Total Stock Market Index Fund ETF)**: A proxy for the overall stock market.
- **DBC (Invesco DB Commodity Index Tracking Fund)**: Represents the commodities market.
- **AGG (iShares Core US Aggregate Bond ETF)**: Captures the bond market dynamics.
- **VIX (Chicago Board Options Exchange's CBOE Volatility Index)**: Provides insights into the overall market volatility.

Each of these indices is distinct and serves as an effective representative for different market characteristics, ensuring a multipronged view of the financial landscape. The authors' data range from 2006 - 2020, and their testing period is from 2011 - 2020, while the data I use range from 2006 - 2023, and the testing period I employ is from 2013 - 2023.

### 3.4 Temporal Considerations
Given that this study uses adjusted closing prices, there exists an inherent information lag. To account for this, the output allocations are lagged by a day, ensuring the methodology respects the chronological nature of information flow in financial markets.

## 4. Modeling

All models were compared against a simple equal-weight portfolio and a baseline mean-variance (maximum Sharpe) portfolio.

### 4.1 Leveraged vs Non-leveraged Scenarios
Before delving into the models, it's imperative to differentiate between leveraged and non-leveraged scenarios:

Leveraged Scenario: Allows borrowing of funds to amplify potential returns on an investment. However, while this offers the potential for magnified profits, it also brings an increased risk of losses.

Non-leveraged Scenario: The more orthodox investment approach, where no borrowing is involved. The potential for returns is directly tied to the initial investment amount, with no amplification. The risk is ostensibly lower compared to the leveraged scenario.

From an architectural perspective, the differentiation between these scenarios is encapsulated in the model's output layer. In a non-leveraged setting, a softmax layer ensures portfolio allocations sum to 1. In contrast, the leveraged scenario employs a sigmoid layer, offering a more flexible allocation scheme where both borrowed and invested funds can be modeled.

### 4.2 Activation Functions: Sigmoid vs Softmax
Understanding the mathematical properties of activation functions is paramount when discerning their suitability for specific tasks, particularly in portfolio management.

#### 4.2.1 Softmax

The Softmax function for a given vector $z$ is defined as:

<p align="center">
    <img src="workflow/img/softmax_func.png" alt="Softmax Function" width="22%" height="22%">
</p>

Where $K$ is the number of classes (or assets in the context of portfolio allocation). The output is a probability distribution over $K$ classes, meaning the sum of the outputs is exactly 1. This is suitable for non-leveraged portfolio allocations where the sum of asset allocations must equal the total investment.

The Softmax function ensures that an increase in the allocation of one asset leads to a corresponding decrease in the allocation of at least one other asset. This is coherent with the non-leveraged scenario, where any increase in one asset's allocation must be offset by reducing others to maintain the total investment constant.

#### 4.2.2 Sigmoid

<p align="center">
    <img src="workflow/img/sigmoid_func.png" alt="Sigmoid Function" width="15%" height="15%">
</p>

The sigmoid function ranges between 0 and 1, and it effectively squashes its input into this interval. This makes it apt for representing probabilities or portfolio allocations when considering leveraged scenarios. A value close to 1 can represent a high allocation to a particular asset, while a value close to 0 can indicate minimal or no allocation.

However, the sigmoid function treats each output independently, meaning that the sum of the outputs is not constrained to be 1. In a leveraged scenario, this flexibility is advantageous, allowing the sum of allocations to be greater than the initial investment (representing borrowing) or less than the initial investment (indicating holding cash).

#### 4.2.3 Interpreting Probabilities as Portfolio Allocations:

The beauty of using these activation functions in portfolio management lies in their ability to transmute raw model outputs into interpretable portfolio allocations. Both sigmoid and softmax render outputs that align with the structure of probability distributions. Within the investment milieu, such probabilities are construed as the proportionate allocations of a portfolio to varying assets. For instance, should an asset procure a probability (or output) of 0.2 from the model, it signifies a recommendation that 20% of the portfolio be assigned to that particular asset. However, neural networks are domain agnostic and don't possess inherent financial cognizance. Its outputs, derived from patterns in the training data, are interpreted in this case by financial experts in the context of asset allocations. The network's main goal aligns with the training objective function (explored in section [4.4](#44-objective-function)), and the specific allocations are but a means to achieve that end. Thus, while the model might not fathom the intricacies of finance, its outputs, when correctly interpreted, can seamlessly integrate into portfolio management.

#### 4.2.4 Leverage Factor

In the non-leveraged approach, softmax normalization ensures that asset allocations sum to 1, effectively capping total portfolio exposure at 100%. In contrast, the leveraged model employs a sigmoid function for each asset, allowing individual allocations to reach up to 1. Consequently, the portfolio's total exposure can amplify to a maximum of 400%, or four times the available capital, when accounting for all four market indices.

### 4.3 Model Architectures, Inputs, Targets, and Outputs

The feature set includes 4 assets (number of indices) and 8 features (4 price series + 4 returns series), a lookback period of 50 days, and a total of 4403 samples (4453 data points - 50 day lookback period). All targets take the form $(4403,\text{ }4)$, representing the daily returns of each market index beyond the first 50 days.

1. **1D Convolutional Neural Network (CNN)**: This model deploys 1D convolutional layers for feature extraction. After the convolution, a dynamic fully connected layer transforms the output for portfolio allocations, which are then passed through a sigmoid activation function (for leverage) or a softmax activation function (no leverage) to ensure values between 0 and 1. `Pytorch`'s `Conv1d` takes an input shape of $(\text{n. samples, n. features, sequence length})$, so for our dataset it would have the shape $(4403,\text{ }8,\text{ }50)$.

2. **Fully Connected Neural Network (FCN)**: Being a more traditional architecture, it relies on two dense layers for processing. The final layer's outputs represent portfolio allocations, post sigmoid / softmax activation, ensuring values remain in the [0, 1] interval. I constructed the FCN via `PyTorch`'s `Linear` layers, which take inputs of size $(\text{n. samples}, \text{ sequence length} \times \text{n. features})$, so for our dataset it would have the shape $(4403,\text{ }50 \times 8)$ or $(4403,\text{ }400)$.

3. **Long Short-Term Memory (LSTM)**: Leveraging the prowess of LSTM  units, this model is designed to recognize temporal patterns in data sequences. The LSTM layer processes sequences, and its outputs are directed to a fully connected layer. The final portfolio allocations, as in other leveraged models, are determined post a sigmoid / softmax activation. `PyTorch`'s `LSTM` class accepts an input of shape $(\text{n. samples, sequence length, n. features})$, so for our dataset it would have the shape $(4403,\text{ }50,\text{ }8)$.

The crux of the optimization problem resides in the above models' output layer, which output 4 allocation values (or portfolio weights), one for each asset. Feel free to peruse the [`no_leverage_models`](workflow/tools/no_leverage_models) and [`leverage_models`](workflow/tools/leverage_models) modules to see the source code for each model and its architecture.

### 4.4 Objective Function

The objective function used to train each model is the key innovation brought forth by the authors of the original paper. Instead of traditional loss functions such as Mean Squared Error (MSE) and the like, each model is trained to maximize the Sharpe ratio per trading period (1 day):

<p align="center">
    <img src="workflow/img/obj_func.png" alt="Objective Function" width="32%" height="32%">
</p>

$R_{p,t}$ is the return of the portfolio at time $t$, taken as the sum product of the allocation at time $t$ determined by the model of each asset $i$ and the asset's actual logarithmic return at time $t$. The expected portfolio return, ${E[R_{p,t}]}$ is taken as the mean of the returns within the batch.

By default, the models follow gradient descent, but we intend to maximize the Sharpe ratio via gradient *ascent*,

<p align="center">
    <img src="workflow/img/gradient_ascent.png" alt="Gradient Ascent" width="15%" height="15%">
</p>

so in the code, the objective function is written to return a negative value.* View the [`metrics`](workflow/tools/metrics.py) module to see the implementation of the objective function in code.

**Note that `PyTorch`'s Adam optimizer now permits a `maximize` parameter so as to eliminate the need to create negative objective functions. However, for the sake of illustration I adhere to the default parameter.*

### 4.5 Training Scheme

The models are retrained every two years, using all data available up until that point. A hidden dimension size of 64 is used, as per the authors' configuration. A validation size of 20% is used, and each testing period spans 504 days (2 years of trading days). For example:

<p align="center">
    <img src="workflow/img/training_scheme.png" alt="Training Scheme" width="55%" height="55%">
    <br>
</p>

### 5. Results

#### 5.1 Performance of Models Over Time

<p align="center">
    <img src="workflow/img/backtest_charts.png" alt="Backtest Charts" width="90%" height="90%">
    <br>
    <i>Figure 1: Equity curves vs. VTI</i>
</p>

In assessing the models over time, the leveraged LSTM model conspicuously stands out with its stellar performance, markedly surpassing its no-leverage counterpart and the broader stock market epitomized by VTI. Such dominance accentuates its adeptness at capturing market trends, capitalizing often antithetical market trends, and overall its prowess in return generation. In juxtaposition, the Mean Variance models, irrespective of whether they deploy leverage or not, tread similar paths, indicating a degree of consistency. Yet, they remain eclipsed by the superior performance of the LSTM models.

Conversely, the FCN and CNN models, both with and without the leverage mechanism, seem to grapple in keeping pace, lagging discernibly behind their LSTM and Mean Variance counterparts. This dichotomy underscores the variability in model efficacies and their intrinsic methodologies.

#### 5.2 Portfolio Metrics

- **Sharpe Ratio**: The leveraged LSTM model yields the highest Sharpe ratio at **1.109**, signifying optimal risk-adjusted returns. This is closely trailed by the no-leverage LSTM model at **1.044**. The stock market's Sharpe ratio stands at **0.597**, with most models, except for the CNN and FCN variants, surpassing this benchmark.

- **Cumulative Returns**: Once again, the leveraged LSTM model reigns supreme with a return of **2.561**. Notably, its non-leveraged counterpart produces a return of **1.674**, underscoring the benefit of leverage in enhancing returns.

- **Annual Volatility**: The leveraged CNN model exhibits the highest volatility at **0.784**, indicative of higher risk. In contrast, the leveraged Mean Variance model manifests the lowest volatility, making it the most stable among the lot.

- **Maximum Drawdown**: All models and the stock market have experienced negative drawdowns. The leveraged CNN model faced the steepest decline with a drawdown of **-1.148**, highlighting potential vulnerabilities.

- **Alpha**: The leveraged LSTM model boasts the highest alpha value at **0.149**, suggesting its ability to generate excess returns compared to the stock market. Most models display positive alpha values, except for the CNN and FCN models without leverage and the Mean Variance models.

In summation, the LSTM models, especially when leveraged, emerge as the most efficacious in this assessment, outpacing other models and the stock market. However, the feasibility of the leveraged models come down to the risk appetite of the investor, given the inherent risks associated with leveraging.

#### 5.3 Performance During the COVID-19 Downturn

As did the authors of the original paper, we can go further in our analysis and see the allocations that led to the stellar performance within each LSTM model. To do this we'll examine the allocations at the COVID-19 downturn, taking place in the first quarter of 2020.

##### 5.3.1 No-leverage LSTM

<p align="center">
    <img src="workflow/img/unlev_lstm_allocations_covid.png" alt="No-Leverage LSTM Allocations, COVID-19" width="50%" height="50%">
    <br>
    <i>Figure 2: No-Leverage LSTM Allocations, Q1 2020</i>
</p>

- **Bonds**: Bond allocation in this variant exhibits a distinct trend. Unlike its leveraged counterpart, the no-leverage LSTM exhibits a steeper incline in bond allocations starting early in the quarter, positioning defensively well before the pronounced upheaval. This sugggests an innate aversion to undue risks.
- **Commodities**: The model's commodities strategy showcases cautious oscillation. Instead of the pronounced flux observed in the Leveraged model, here the allocations are restrained, which is more circumspect in nature.
- **Stocks**: The model mitigated exposures to stocks and instead loaded up on bonds during the late February to mid-March window, around when the stock market crash began. It began to increase its stock holdings thereafter once, in time for the bull market that followed.
- **Volatility**: In the face of burgeoning volatility, the model minimized its volatility allocations when volatility surges are anticipated, revealing its conservative stance.

##### 5.3.2 Leveraged LSTM

<p align="center">
    <img src="workflow/img/lev_lstm_allocations_covid.png" alt="Leveraged LSTM Allocations, COVID-19" width="50%" height="50%">
    <br>
    <i>Figure 3: Leveraged LSTM Allocations, Q1 2020</i>
</p>

***Figure 3*** demarcates the astuteness of the LSTM model's asset allocation acumen amidst the turbulent outset of 2020:

- **Bonds**: While bond prices demonstrated moderate volatility, the LSTM model maintains a stable allocation in bonds, predominantly oscillating between 86% to 89%. A notable spike in bond allocation circa March aligns with a temporary ascent in bond prices, hinting at both the model's ability to capitalize on upward price movements and its repositioning toward safer instruments prior to a stark downturn.
- **Commodities**: The allocation for commodities undergoes pronounced flux as commodity prices oscillate with broader amplitude; the model's allocations seem to somewaht mirror these shifts.
- **Stocks**: The model seems to have anticipated the impending crash starting March, where it reduced stock allocations from approximately 80% to just over 65%.
- **Volatility**: Even as volatility prices soar dramatically mid-quarter, the LSTM model contracts its allocation, reflecting its anticipatory stance to sidestep the ensuing volatility surge.