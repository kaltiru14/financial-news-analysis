# Financial News Analysis - Notebooks

This directory contains Jupyter notebooks for the three main tasks of the financial news analysis project.

## 📁 Notebook Files

### 1. `task1_eda.ipynb` - Exploratory Data Analysis
**Purpose**: Comprehensive exploration of the financial news dataset

#### Analysis Performed:
- **Data Loading & Assessment**
  - Dataset overview and basic statistics
  - Missing value analysis
  - Data type validation

- **Descriptive Statistics**
  - Headline text analysis (length, word count)
  - Publisher distribution and activity analysis
  - Basic data quality assessment

- **Temporal Analysis**
  - Publication date range analysis
  - Hourly and daily publication patterns
  - Time series trends in news volume

- **Text Analysis**
  - Common keywords and phrases in headlines
  - Financial terminology frequency analysis
  - Topic modeling preparation

- **Publisher Analysis**
  - Top publishers by article volume
  - Publisher domain analysis (if emails present)
  - Source concentration analysis

- **Visualizations**
  - Headline length distribution
  - Top publishers bar chart
  - Publication frequency by hour/day
  - Temporal trends over time
  - Word count distribution

#### Key Outputs:
- Data quality report
- Publication pattern insights
- Publisher concentration analysis
- Professional visualizations in `/images` folder

---

### 2. `task2_technical_analysis.ipynb` - Quantitative Analysis with TA-Lib
**Purpose**: Technical analysis of stock price data using TA-Lib indicators

#### Analysis Performed:
- **Data Preparation**
  - Load stock price data from CSV files
  - Validate required columns (Open, High, Low, Close, Volume)
  - Date formatting and indexing

- **TA-Lib Technical Indicators**
  - Moving Averages (SMA_20, SMA_50, EMA_12, EMA_26)
  - Relative Strength Index (RSI_14)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - Average True Range (ATR)
  - On Balance Volume (OBV)

- **Financial Metrics (PyNance-like)**
  - Daily and cumulative returns
  - Volatility calculations (20-day rolling)
  - Sharpe ratio (simplified)
  - Support and resistance levels
  - Price momentum signals

- **Multi-Stock Analysis**
  - Comparative analysis across multiple stocks
  - Performance metrics comparison
  - Trading signals generation

- **Visualizations**
  - Price and moving averages with Bollinger Bands
  - RSI with overbought/oversold levels
  - MACD indicator plots
  - Volume and OBV analysis
  - Multi-stock comparison charts

#### Stocks Analyzed:
- AAPL, AMZN, GOOG, META, MSFT, NVDA

#### Key Outputs:
- Comprehensive technical indicators for each stock
- Trading signals and momentum analysis
- Professional technical analysis charts
- Multi-stock performance comparison

---

### 3. `task3_correlation_analysis.ipynb` - Correlation between News and Stock Movements
**Purpose**: Analyze correlation between news sentiment and stock price movements

#### Analysis Performed:
- **Data Alignment**
  - Date normalization between news and stock datasets
  - Alignment of news articles with corresponding trading days
  - Overlapping date analysis

- **Sentiment Analysis**
  - Combined TextBlob + NLTK VADER sentiment scoring
  - Sentiment classification (positive, negative, neutral)
  - Daily sentiment aggregation
  - Publisher-level sentiment analysis

- **Stock Movement Analysis**
  - Daily percentage return calculations
  - Cumulative return analysis
  - Volatility measurements
  - Performance metrics

- **Correlation Analysis**
  - Pearson correlation coefficient
  - Spearman rank correlation
  - Statistical significance testing (p-values)
  - Correlation strength interpretation

- **Multi-Stock Comparative Analysis**
  - Cross-stock correlation comparison
  - News coverage effectiveness analysis
  - Best-performing sentiment predictors

- **Business Insights**
  - Trading strategy recommendations
  - Sentiment-based prediction potential
  - Risk management considerations

#### Object-Oriented Implementation:
The analysis uses a modular OO architecture:
- `DataLoader`: Data loading and validation
- `SentimentAnalyzer`: TextBlob + VADER sentiment analysis
- `StockAnalyzer`: Return calculations and metrics
- `CorrelationEngine`: Main analysis pipeline

#### Key Outputs:
- Individual stock correlation reports
- Statistical significance analysis
- Professional correlation visualizations
- Business insights and recommendations
- Comparative analysis across all stocks

---

## 🛠 Technical Requirements

### Python Libraries Used:
- **Core**: pandas, numpy, matplotlib, seaborn
- **NLP**: nltk, textblob
- **Financial Analysis**: TA-Lib
- **Statistics**: scipy, scikit-learn
- **Visualization**: matplotlib, seaborn

### Data Files Required:
- `../data/raw_analyst_ratings.csv` - Financial news dataset
- `../data/[SYMBOL].csv` - Stock price data (AAPL, AMZN, etc.)

### Generated Outputs:
- Visualizations saved to `../images/` folder
- Analysis reports and business insights
- Statistical correlation results

---

## 🚀 Execution Order

1. **Start with Task 1** (`task1_eda.ipynb`) to understand the data
2. **Proceed to Task 2** (`task2_technical_analysis.ipynb`) for technical indicators
3. **Complete with Task 3** (`task3_correlation_analysis.ipynb`) for sentiment-stock correlation

---

## 📊 Success Metrics

### Task 1 - EDA
- ✅ Comprehensive data understanding
- ✅ Identification of data patterns and issues
- ✅ Professional visualizations
- ✅ Actionable insights for subsequent analysis

### Task 2 - Technical Analysis
- ✅ Accurate TA-Lib indicator calculations
- ✅ Meaningful trading signals
- ✅ Professional financial charts
- ✅ Multi-stock comparison

### Task 3 - Correlation Analysis
- ✅ Robust date alignment
- ✅ Accurate sentiment scoring
- ✅ Statistical correlation analysis
- ✅ Business-ready insights and recommendations

---

## 🔧 Troubleshooting

### Common Issues:
1. **Missing data files**: Ensure all CSV files are in the `../data/` directory
2. **Library imports**: Run `pip install -r ../requirements.txt` first
3. **TA-Lib installation**: May require specific installation on Windows
4. **Date parsing issues**: Check date formats in source CSV files

### Support:
- Check individual notebook cells for specific error handling
- Verify file paths and data availability
- Ensure all dependencies are installed

---

## 📈 Business Value

This analysis provides:
- **Market intelligence** from news sentiment patterns
- **Trading signals** from technical indicators
- **Predictive insights** from sentiment-price correlations
- **Risk management** through comprehensive analysis

**Completion Status**: All three tasks successfully implemented and tested.