# Retail Analytics Suite

# Project Overview
The Retail Analytics Suite is a comprehensive data analysis and visualization platform that combines sales data from multiple retail sources to provide actionable insights and predictive analytics. The application analyzes Walmart sales data, Amazon sales data, and Amazon customer sentiment data to deliver a holistic view of retail performance and customer satisfaction.

# Data Sources
1. Walmart Sales Data

- Source: Walmart store sales records
- Period: 2020-2021
- Format: CSV
- Update Frequency: Weekly
- Key metrics: Weekly sales, store information, and economic indicators
- Link : https://www.kaggle.com/datasets/yasserh/walmart-dataset
 
2. Amazon Sales Data FY2020-21

- Source: Amazon sales records
- Period: 2020-2021 fiscal year
- Format: CSV
- Update Frequency: Daily
- Key metrics: Order details, product information, and pricing data
- Link : https://www.kaggle.com/datasets/earthfromtop/amazon-sales-fy202021

3. Amazon Customer Reviews

- Source: Amazon customer feedback
- Period: 2020-2021
- Format: CSV
- Update Frequency: Daily
- Key metrics: Review content, ratings, and customer sentiment
- Link : https://www.kaggle.com/datasets/ahmedsayed564/amazon-sales-dataset

# Features
1. Sales Analysis

- Time series visualization
- Seasonal trend analysis
- Store performance comparison
- Revenue forecasting

2. Customer Sentiment Analysis

- Review sentiment scoring
- Word cloud visualization
- Category-wise sentiment analysis
- Trend analysis of customer satisfaction

3. Price Analytics

- Price distribution analysis
- Discount impact assessment
- Category-wise pricing strategies
- Competitive pricing analysis

4. Interactive Dashboards

- Dynamic filtering
- Real-time calculations
- Custom date range selection
- Multiple visualization options

# Modeling Approach
### 1. Sales Forecasting Models
### Prophet Model

- Purpose: Long-term sales forecasting with seasonal decomposition
- Implementation:

- Multiplicative seasonality model
- Yearly, weekly, and daily seasonality components
- Holiday effects incorporation
- Automatic changepoint detection


 ### Advantages:

- Handles missing data effectively
- Captures multiple seasonality patterns
- Robust to outliers
- Automatic trend changepoint detection



# SARIMA Model

- Purpose: Short to medium-term sales forecasting
  Implementation:

- Order selection: (1,1,1) for trend
- Seasonal order: (1,1,1,7) for weekly patterns
- Rolling validation approach

### Advantages:

- Captures both trend and seasonality
- Provides confidence intervals
- Strong statistical foundation
- Good for short-term predictions



# Random Forest Regressor

- Purpose: Feature-based sales prediction
  Implementation:

- 200 trees with max depth of 15
- Minimum samples split: 5
- Minimum samples leaf: 2

### Advantages:

- Handles non-linear relationships
- Feature importance ranking
- Robust to outliers
- Good for complex pattern recognition



### 2. Sentiment Analysis Model
# NLTK Sentiment Analyzer

- Purpose: Customer review sentiment scoring
  Implementation:

- VADER sentiment scoring
- Custom text preprocessing
- Weighted sentence analysis

### Advantages:

- Specifically tuned for social media text
- Handles emoji and punctuation
- Multiple sentiment dimensions
