# Retail Analytics Suite

# Table of Contents

1. Project Overview

## Project Overview
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

# Installation & Setup

# Clone the repository
git clone 'https://github.com/Diparna/Sales_Forecasting/'

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Usage Guide

### Data Upload:

Format requirements
Supported file types


### Making Predictions:

Step-by-step guide
Input parameters explanation

### Interpreting Results:

Understanding the visualizations
Reading the forecasts

# Real-World Applications
### 1. Retail and E-Commerce:

Predict demand for products to optimize inventory and avoid stockouts or overstocking.
Plan promotional campaigns based on forecasted sales trends.

### 2.Supply Chain Management:

Anticipate demand spikes to adjust production schedules and logistics planning.
Improve supplier relations by sharing accurate demand forecasts.

### 3.Finance and Budgeting:

Help businesses allocate budgets effectively by forecasting revenue streams.
Aid in financial planning for startups or small businesses.

### 4.Hospitality and Tourism:

Predict seasonal trends in bookings, helping with staff planning and resource allocation.
Forecast customer preferences for targeted marketing.

### 5.Food and Beverage Industry:

Assist restaurants and cafes in purchasing raw materials by forecasting sales volumes.
Predict trends in customer dining habits for menu optimization.

6.Manufacturing:

Plan production schedules based on sales forecasting to reduce waste.
Align resources with expected demand for better cost management.

# Educational Applications

### 1. Teaching Data Science Concepts:

Demonstrate time-series analysis, ARIMA models, and machine learning applications in forecasting.
Visualize the impact of data preprocessing and model selection on predictive accuracy.

### 2. Interactive Learning for Students:

Allow students to upload their datasets to practice forecasting in real-world scenarios.
Provide hands-on experience with sales datasets for project-based learning.

### 3. Museum or Science Fair Exhibits:

Illustrate the power of predictive analytics in a simple, user-friendly manner.
Engage visitors by allowing them to tweak parameters and see how forecasts change.

### 4. Business School Case Studies:

Use the app as a sandbox environment for students to explore retail analytics.
Simulate scenarios to teach resource allocation based on demand predictions.

### 5. Coding Workshops:

Serve as a tool for teaching Streamlit, Python, and data visualization techniques.
Show how to integrate data science with app development to build impactful tools.
