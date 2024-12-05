import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import nltk

nltk.download('vader_lexicon')
nltk.download('punkt_tab')

st.set_page_config(
    page_title="Retail Analytics Suite",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Retail Analytics Suite")
    
    # Add app selection to sidebar
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.radio(
        "Choose Analysis",
        ["Home", "Walmart Sales Analysis", "Amazon Sales Prediction", "Amazon Sentiment Analysis"]
    )
    
    if app_selection == "Home":
        show_home()
    elif app_selection == "Walmart Sales Analysis":
        walmart_app()
    elif app_selection == "Amazon Sales Prediction":
        amazon_sales_app()
    else:
        amazon_sentiment_app()

    # Add a vertical spacer
    st.sidebar.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
    
    # Create a fixed container at the bottom of the sidebar
    st.sidebar.markdown(
    """
    <div style='position: fixed; bottom: 0; left: 0; padding: 1rem; background-color: #0E1117; width: 17rem;'>
        <h2 style='font-size: 1.25rem; margin-bottom: 0.5rem;'>GitHub Repository 🤖</h2>
        <a href='https://github.com/Diparna/Sales_Forecasting' target='_blank' style='color: #4B9DFF; text-decoration: none;'>
            View the source code on GitHub 💻
        </a>
        <p style='margin-top: 0.5rem; font-size: 0.875rem;'>Made by Diparna Adhikary for educational purpose! 📖</p>
    </div>
    """,
    unsafe_allow_html=True
)

    

def show_home():
    st.markdown("""
    ## Welcome to the Retail Analytics Suite! 📊
    
    This comprehensive suite combines three powerful analysis tools:
    
    ### 1. Walmart Sales Analysis 🏪
    - For Users, Please go to 'Store Timeline' and 'Sales Forecasting'
    - To see what's going on under the hood, Please use any of the other pages!
    
    ### 2. Amazon Sales Prediction 📈
    - For Users, Please go to 'Sales Prediction'
    - To see what's going on under the hood, Please use any of the other pages!
    
    ### 3. Amazon Sentiment Analysis 💭
    - For Users, Please go to 'Product Recommendations'
    - To see what's going on under the hood, Please use any of the other pages!
    
    Use the radio buttons in the sidebar to navigate between different analyses.
    """)

def walmart_app():
    # Clear any existing Streamlit state
    st.empty()
    
    # Run the Walmart analysis app
    @st.cache_data  # Cache the data loading to improve performance
    def load_data():
        url = "https://raw.githubusercontent.com/Diparna/Sales_Forecasting/refs/heads/main/Walmart_Sales_Data.csv"
        df = pd.read_csv(url)
        df['Date'] = df['Date'].apply(parse_mixed_dates)
        return df
        
    def parse_mixed_dates(date_string):
        try:
            return pd.to_datetime(date_string, format='%d-%m-%Y')
        except:
            try:
                return pd.to_datetime(date_string, format='%m/%d/%Y')
            except:
                return pd.NaT

    def prepare_data_for_prophet(df):
        """Prepare data for Prophet model"""
        # Group by date and calculate total sales
        daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        
        # Rename columns to Prophet requirements
        prophet_df = daily_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        return prophet_df

    def train_prophet_model(df, seasonality_mode='multiplicative'):
        """Train Prophet model with specified parameters"""
        # Initialize model
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add additional seasonality if needed
        model.add_country_holidays(country_name='US')

        # Fit model
        model.fit(df)
        return model

    def make_future_predictions(model, periods=52):
        """Generate future predictions"""
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='W')
        
        # Make predictions
        forecast = model.predict(future)
        return forecast

    def plot_forecast(original_data, forecast_data):
        """Create interactive forecast plot"""
        fig = go.Figure()

        # Plot original data
        fig.add_trace(go.Scatter(
            x=original_data['ds'],
            y=original_data['y'],
            name='Historical Sales',
            mode='lines',
            line=dict(color='blue')
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            name='Forecast',
            mode='lines',
            line=dict(color='red')
        ))

        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
            y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        fig.update_layout(
            title='Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        return fig
    def plot_components(model, forecast):
        """Plot trend and seasonality components"""
        # Trend
        trend_fig = px.line(
            forecast, 
            x='ds', 
            y='trend',
            title='Sales Trend'
        )
        
        # Weekly seasonality
        weekly_seasonality = model.weekly_seasonality
        if weekly_seasonality:
            weekly_df = pd.DataFrame({
                'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'Value': forecast['weekly'].iloc[0:7]  # Get the first week's pattern
            })
            weekly_fig = px.line(
                weekly_df,
                x='Day',
                y='Value',
                title='Weekly Seasonality Pattern'
            )
        else:
            weekly_fig = None
            
        return trend_fig, weekly_fig

    def evaluate_forecast(model, df):
        """Calculate forecast evaluation metrics"""
        # Make predictions for training data
        predictions = model.predict(df)
        
        # Calculate metrics
        mae = mean_absolute_error(df['y'], predictions['yhat'])
        mse = mean_squared_error(df['y'], predictions['yhat'])
        rmse = np.sqrt(mse)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

    def sales_forecasting_page():
        st.header('Walmart Sales Forecasting')
        
        # Load data
        @st.cache_data
        def load_data():
            url = "https://raw.githubusercontent.com/Diparna/Sales_Forecasting/refs/heads/main/Walmart_Sales_Data.csv"
            df = pd.read_csv(url)
            df['Date'] = df['Date'].apply(parse_mixed_dates)
            return df
        
        df = load_data()
        
        # Data preparation
        prophet_df = prepare_data_for_prophet(df)
        
        # Add detailed descriptions for seasonality modes
        seasonality_help = """
        Choose how seasonal components combine with the trend:
        
        • Multiplicative: Use when the seasonal variations increase with the trend
            - Best for data where seasonal fluctuations grow larger as the trend increases
            - Example: If summer sales are typically 20% higher than average, they'll be 
            proportionally higher when overall sales are higher
            - Good for retail data where seasonal spikes grow with overall business growth
        
        • Additive: Use when the seasonal variations are consistent regardless of the trend
            - Best for data where seasonal fluctuations remain relatively constant
            - Example: If summer typically adds $1M to sales, it will add the same amount 
            regardless of the overall trend
            - Better for data where seasonal effects are more stable over time
        """

        # Model parameters
        st.subheader('Model Configuration')
        seasonality_mode = st.selectbox(
            'Seasonality Mode',
            options=['multiplicative', 'additive'],
            help=seasonality_help
        )
        
        forecast_periods = st.slider(
            'Forecast Periods (weeks)',
            min_value=4,
            max_value=52,
            value=12,
            help='Number of weeks to forecast into the future'
        )
        
        # Train model
        with st.spinner('Training forecasting model...'):
            model = train_prophet_model(prophet_df, seasonality_mode)
            forecast = make_future_predictions(model, periods=forecast_periods)
        
        # Display forecast plot
        st.subheader('Sales Forecast')
        forecast_plot = plot_forecast(prophet_df, forecast)
        st.plotly_chart(forecast_plot, use_container_width=True)
        
        # Model components
        st.subheader('Forecast Components')
        trend_fig, weekly_fig = plot_components(model, forecast)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(trend_fig, use_container_width=True)
        with col2:
            if weekly_fig:
                st.plotly_chart(weekly_fig, use_container_width=True)
            else:
                st.write("Weekly seasonality pattern not available")

        # Model evaluation
        st.subheader('Model Performance')
        metrics = evaluate_forecast(model, prophet_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Mean Absolute Error', f"${metrics['MAE']/1000000:,.2f}M")
        with col2:
            st.metric('Mean Squared Error', f"${metrics['MSE']/1000000:,.2f}M")
        with col3:
            st.metric('Root Mean Squared Error', f"${metrics['RMSE']/1000000:,.2f}M")
        
        # Download forecast
        st.subheader('Download Forecast')
        forecast_download = pd.DataFrame({
            'Date': forecast['ds'],
            'Predicted_Sales': forecast['yhat'],
            'Lower_Bound': forecast['yhat_lower'],
            'Upper_Bound': forecast['yhat_upper']
        })
        
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_download.to_csv(index=False),
            file_name="walmart_sales_forecast.csv",
            mime="text/csv"
        )

    def feature_engineering(df):
        df_engineered = df.copy()
        
        # Temporal Features
        df_engineered['Day_of_Week'] = df_engineered['Date'].dt.day_name()
        df_engineered['Month'] = df_engineered['Date'].dt.month
        df_engineered['Quarter'] = df_engineered['Date'].dt.quarter
        df_engineered['Is_Weekend'] = df_engineered['Date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
        
        # Seasonal Indicators
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        df_engineered['Season'] = df_engineered['Date'].dt.month.apply(get_season)
        
        # Sales-related Features
        df_engineered['Sales_Log'] = np.log1p(df_engineered['Weekly_Sales'])
        
        return df_engineered

    def prepare_data_for_arima(df):
        """Prepare data for ARIMA model"""
        # Group by date and calculate total sales
        daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        # Set date as index
        daily_sales.set_index('Date', inplace=True)
        return daily_sales

    def train_arima_model(data, ar_order=1):
        """Train SARIMA model with specified parameters"""
        model = SARIMAX(
            data,
            order=(ar_order, 1, 0),  # (p, d, q) - simplified to just AR and differencing
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit()
        return results

    def plot_arima_forecast(data, forecast, forecast_dates):
        """Create interactive ARIMA forecast plot"""
        fig = go.Figure()

        # Plot original data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Weekly_Sales'],
            name='Historical Sales',
            mode='lines',
            line=dict(color='blue')
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            name='ARIMA Forecast',
            mode='lines',
            line=dict(color='red')
        ))

        fig.update_layout(
            title='ARIMA Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        return fig

    def plot_residuals(model):
        """Plot simplified residuals analysis using separate plots"""
        # Create a DataFrame with the residuals
        residuals_df = pd.DataFrame({
            'Date': model.data.dates,
            'Residual': model.resid
        })
        
                
        # Residual distribution
        residuals_dist = px.histogram(
            residuals_df,
            x='Residual',
            title='Residual Distribution',
            nbins=30
        )
        residuals_dist.update_layout(
            xaxis_title='Residual Value',
            yaxis_title='Count'
        )
        
        return residuals_dist

    def arima_forecasting_page():
        st.header('Walmart ARIMA Sales Forecasting')
        
        # Load and prepare data
        df = load_data()
        sales_data = prepare_data_for_arima(df)
        
        # Simplified model parameters
        st.subheader('Model Configuration')
        col1, col2 = st.columns(2)
        
        with col1:
            ar_order = st.number_input('AR Order (p)', min_value=1, max_value=5, value=1,
                                    help='Order of the AutoRegressive component')
        
        with col2:
            forecast_steps = st.number_input('Forecast Steps (weeks)', 
                                        min_value=1, max_value=52, value=12,
                                        help='Number of weeks to forecast')
        
        # Train model
        with st.spinner('Training ARIMA model...'):
            try:
                model = train_arima_model(sales_data['Weekly_Sales'], ar_order=ar_order)
                
                # Generate forecast
                forecast = model.get_forecast(steps=forecast_steps)
                forecast_mean = forecast.predicted_mean
                
                # Create forecast dates
                last_date = sales_data.index[-1]
                forecast_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='W')[1:]
                
                # Plot forecast
                st.subheader('Sales Forecast')
                forecast_fig = plot_arima_forecast(sales_data, forecast_mean, forecast_dates)
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Model diagnostics
                st.subheader('Model Diagnostics')
                
                # Show model summary in expander
                with st.expander("View Model Summary"):
                    st.text(str(model.summary()))
                
                # Plot residuals analysis using separate plots
                st.subheader('Residuals Analysis')
                residuals_dist = plot_residuals(model)
                
                st.plotly_chart(residuals_dist, use_container_width=True)
                
                # Add interpretation of residuals
                st.write("""
                **Interpreting the Residuals:**
                - The residual distribution should ideally be centered around zero and roughly normal
                """)
                
                # Model metrics
                st.subheader('Model Performance')
                mae = mean_absolute_error(sales_data['Weekly_Sales'][-forecast_steps:], 
                                        model.get_prediction(start=-forecast_steps).predicted_mean)
                mse = mean_squared_error(sales_data['Weekly_Sales'][-forecast_steps:], 
                                    model.get_prediction(start=-forecast_steps).predicted_mean)
                rmse = np.sqrt(mse)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('MAE', f"${mae/1000000:,.2f}M")
                with col2:
                    st.metric('MSE', f"${mse/1000000:,.2f}M")
                with col3:
                    st.metric('RMSE', f"${rmse/1000000:,.2f}M")
                
                # Download forecast
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted_Sales': forecast_mean
                })
                
                st.download_button(
                    label="Download ARIMA Forecast CSV",
                    data=forecast_df.to_csv(index=False),
                    file_name="walmart_arima_forecast.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error in model training: {str(e)}")
                st.write("Try adjusting the AR order or reducing the forecast steps.")

    def train_sarima_model(data, ar_order=1, seasonal_ar=1):
        """Train SARIMA model with seasonal components"""

        model = SARIMAX(
            data,
            order=(ar_order, 1, 0),
            seasonal_order=(seasonal_ar, 1, 0, 52),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit()
        return results

    def plot_sarima_forecast(data, forecast, forecast_dates):
        """Create interactive SARIMA forecast plot with improved styling"""
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Weekly_Sales'],
            name='Historical Sales',
            mode='lines',
            line=dict(color='blue', width=1)
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            name='SARIMA Forecast',
            mode='lines',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title='SARIMA Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x unified',
            showlegend=True,
            template='plotly_dark',  # Using dark theme
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def sarima_forecasting_page():
        st.header('Walmart SARIMA Sales Forecasting')
        
        # Load and prepare data
        df = load_data()
        sales_data = prepare_data_for_arima(df)
        
        # Simplified model parameters
        st.subheader('Model Configuration')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ar_order = st.number_input('AR Order (p)', 
                                    min_value=1, max_value=5, value=2,
                                    help='Order of the AutoRegressive component')
        
        with col2:
            seasonal_ar = st.number_input('Seasonal AR Order (P)', 
                                        min_value=1, max_value=2, value=1,
                                        help='Order of the seasonal component')
        
        with col3:
            forecast_steps = st.number_input('Forecast Steps (weeks)', 
                                        min_value=1, max_value=52, value=12,
                                        help='Number of weeks to forecast')
        
        # Train model
        with st.spinner('Training SARIMA model...'):
            try:
                model = train_sarima_model(sales_data['Weekly_Sales'], 
                                        ar_order=ar_order,
                                        seasonal_ar=seasonal_ar)
                
                # Generate forecast
                forecast = model.get_forecast(steps=forecast_steps)
                forecast_mean = forecast.predicted_mean
                
                # Create forecast dates
                last_date = sales_data.index[-1]
                forecast_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='W')[1:]
                
                # Plot forecast
                st.subheader('Sales Forecast')
                forecast_fig = plot_sarima_forecast(sales_data, forecast_mean, forecast_dates)
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Model evaluation and metrics
                st.subheader('Model Performance')
                
                # Calculate metrics on the last portion of data (validation)
                validation_steps = min(forecast_steps, 12)  # Use up to 12 weeks for validation
                validation_pred = model.get_prediction(start=-validation_steps)
                validation_mean = validation_pred.predicted_mean
                actual_values = sales_data['Weekly_Sales'][-validation_steps:]
                
                mae = mean_absolute_error(actual_values, validation_mean)
                mse = mean_squared_error(actual_values, validation_mean)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - validation_mean) / actual_values)) * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write('MAE', f"${mae/1000000:,.2f}M")
                with col2:
                    st.write('MSE', f"${mse/1000000:,.2f}M")
                with col3:
                    st.write('RMSE', f"${rmse/1000000:,.2f}M")
                with col4:
                    st.write('MAPE', f"{mape:.2f}%")
                
                # Model interpretation
                st.subheader('Model Interpretation')
                st.write("""
                **Key Observations:**
                - The model captures both short-term patterns (through AR terms) and yearly seasonality
                - MAPE (Mean Absolute Percentage Error) shows the average percentage deviation of forecasts
                - Lower MAPE values indicate better forecast accuracy
                """)
                
                # Download forecast
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted_Sales': forecast_mean
                })
                
                st.download_button(
                    label="Download SARIMA Forecast CSV",
                    data=forecast_df.to_csv(index=False),
                    file_name="walmart_sarima_forecast.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error in model training: {str(e)}")
                st.write("Try adjusting the model parameters.")


    def create_animated_timeline(store_data):
        """Create an animated timeline of store sales"""
        # Create figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[store_data['Date'].iloc[0]],
                    y=[store_data['Weekly_Sales'].iloc[0]],
                    mode='lines+markers',
                    line=dict(width=2, color='blue'),
                    marker=dict(size=8, color='blue'),
                    name='Weekly Sales'
                )
            ],
            layout=go.Layout(
                xaxis=dict(range=[store_data['Date'].min(), store_data['Date'].max()],
                        title='Date'),
                yaxis=dict(range=[0, store_data['Weekly_Sales'].max() * 1.1],
                        title='Sales ($)'),
                title=f'Sales Timeline Animation',
                showlegend=True,
                template='plotly_dark',
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=1.15,
                        x=1.15,
                        xanchor='right',
                        yanchor='top',
                        pad=dict(t=0, r=10),
                        buttons=[
                            dict(label='Play',
                                method='animate',
                                args=[None, 
                                    dict(frame=dict(duration=100, redraw=True),
                                        fromcurrent=True,
                                        mode='immediate',
                                        transition=dict(duration=50))]),
                            dict(label='Pause',
                                method='animate',
                                args=[[None],
                                    dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate',
                                        transition=dict(duration=0))])
                        ]
                    )
                ]
            ),
            frames=[
                go.Frame(
                    data=[
                        go.Scatter(
                            x=store_data['Date'][:k+1],
                            y=store_data['Weekly_Sales'][:k+1],
                            mode='lines+markers',
                            line=dict(width=2, color='blue'),
                            marker=dict(size=8, color='blue')
                        )
                    ]
                )
                for k in range(1, len(store_data))
            ]
        )
        
        return fig

    def store_timeline_page():
        st.header('Store Sales Timeline Analysis')

        # Load data
        df = load_data()
        
        # Get unique store numbers
        stores = sorted(df['Store'].unique())
        
        # Store selection and controls
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_store = st.selectbox('Select Store:', stores)
            
        # Filter data for selected store
        store_data = df[df['Store'] == selected_store].sort_values('Date')
        
        # Calculate store statistics
        avg_weekly_sales = store_data['Weekly_Sales'].mean()
        max_weekly_sales = store_data['Weekly_Sales'].max()
        min_weekly_sales = store_data['Weekly_Sales'].min()
        total_sales = store_data['Weekly_Sales'].sum()
        
        # Display store metrics
        st.subheader(f'Store {selected_store} Overview')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("Average Weekly Sales", f"${avg_weekly_sales:,.2f}")
        with col2:
            st.write("Maximum Weekly Sales", f"${max_weekly_sales:,.2f}")
        with col3:
            st.write("Minimum Weekly Sales", f"${min_weekly_sales:,.2f}")
        with col4:
            st.write("Total Sales", f"${total_sales:,.2f}")
        
        # Animated Timeline
        st.subheader('Animated Sales Timeline')
        fig = create_animated_timeline(store_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a note about the animation
        st.info("""
        💡 Click the 'Play' button in the top-right corner to start the animation.
        The visualization will show how sales evolved over time.
        Click 'Pause' to stop at any point.
        """)
        
        # Monthly Trends
        st.subheader('Monthly Sales Trends')
        monthly_avg = store_data.set_index('Date')['Weekly_Sales'].resample('M').mean()
        monthly_fig = px.line(
            monthly_avg,
            title=f'Monthly Average Sales',
            labels={'value': 'Average Sales ($)', 'Date': 'Month'}
        )
        monthly_fig.update_layout(template='plotly_dark')
        st.plotly_chart(monthly_fig, use_container_width=True)
        
        # Summary statistics in an expander
        with st.expander("View Detailed Statistics"):
            st.write("**Weekly Sales Statistics:**")
            st.write(store_data['Weekly_Sales'].describe())

    # Load and process data
    with st.spinner('Loading data...'):
        df = load_data()
        st.success('Data loaded successfully!')
    
        # Load data
        with st.spinner('Loading data...'):
            df = load_data()
            
    # Parse dates
    df['Date'] = df['Date'].apply(parse_mixed_dates)
        
    # Perform feature engineering
    df_engineered = feature_engineering(df)
        
    # Sidebar for navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Choose a page', 
        ['Store Timeline',
         'Sales Forecasting',
        'Original Dataset Overview', 
        'Engineered Features Overview', 
        'Feature Distribution', 
        'Correlation Analysis',
        'Seasonal Sales Analysis',
        'ARIMA Forecasting',
        'Summary'
       # 'SARIMA Forecasting',
        ])
        
    if page == 'Original Dataset Overview':
        st.header('Original Dataset Overview')
        st.write("""
        This section presents the raw dataset and its basic characteristics. We can observe the
        initial structure of our data and key statistical measurements.
        """)
        st.info("""
        Key Features:
        • View sample data rows to understand the structure
        • Check dataset dimensions and column descriptions
        • Analyze basic statistical measures of numerical columns
        • Identify missing values and data quality issues
        • Visualize top-performing stores by total sales
        """)
        # Display sample data
        st.subheader('Sample Data')
        st.write("Here's a glimpse of the first few rows of our dataset:")
        st.write(df.head())
        
        # Dataset info
        st.subheader('Dataset Information')
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of Rows: {df.shape[0]:,}")
        with col2:
            st.write(f"Number of Columns: {df.shape[1]}")
            
        st.write("Available Columns:")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")
        
        # Basic statistics
        st.subheader('Statistical Summary')
        st.write("""
        Below are the key statistical measures for numerical columns. This includes count, mean,
        standard deviation, minimum, maximum, and quartile values.
        """)
        st.write(df.describe())
        
        # Missing values analysis
        st.subheader('Missing Values Analysis')
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("Missing values found in the following columns:")
            st.write(missing_data[missing_data > 0])
        else:
            st.write("No missing values found in the dataset!")
        
        # Top 5 Stores Analysis
        st.subheader('Top 5 Stores by Sales')
        top_stores = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(5)

        # Create bar plot for top stores
        fig_top_stores = plt.figure(figsize=(10, 6))
        plt.bar(top_stores.index.astype(str), top_stores.values)
        plt.title('Top 5 Stores by Total Sales')
        plt.xlabel('Store Number')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=0)
        # Format y-axis labels to show millions
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        st.pyplot(fig_top_stores)

        # Display actual numbers
        st.write("Sales breakdown for top performing stores:")
        for store, sales in top_stores.items():
            st.write(f"Store {store}: ${sales:,.2f}")

    elif page == 'Engineered Features Overview':
        st.header('Engineered Features Overview')
        st.write("""
        This section showcases the additional features we've created to enhance our analysis.
        These engineered features help us better understand temporal patterns and seasonal effects
        in the sales data.
        """)
        st.info("""
        Key Features:
        • Temporal features: Day of Week, Month, Quarter
        • Binary indicators: Weekend vs Weekday
        • Seasonal categorization
        • Log-transformed sales for better distribution analysis
        • View the impact of feature engineering on the dataset
        """)
        st.write(df_engineered.head())
        st.write('New Columns:', list(set(df_engineered.columns) - set(df.columns)))
            
        # Display new features
        st.subheader('Newly Created Features')
        new_features = list(set(df_engineered.columns) - set(df.columns))
        st.write("The following features have been engineered from the original data:")
        for feature in new_features:
            st.write(f"- **{feature}**: ", end='')
            if feature == 'Day_of_Week':
                st.write("Day of the week extracted from the date")
            elif feature == 'Month':
                st.write("Month number extracted from the date")
            elif feature == 'Quarter':
                st.write("Quarter of the year (1-4)")
            elif feature == 'Is_Weekend':
                st.write("Binary indicator for weekends (1) vs weekdays (0)")
            elif feature == 'Season':
                st.write("Season categorization (Spring, Summer, Autumn, Winter)")
            elif feature == 'Sales_Log':
                st.write("Log-transformed weekly sales for better distribution")
        
    elif page == 'Feature Distribution':
        st.header('Feature Distribution')
        st.info("""
        Key Features:
        • Interactive histogram plots with density curves
        • Detailed statistical summaries for each feature
        • Select any numerical column to analyze its distribution
        • Compare different features' distributions
        • Identify outliers and unusual patterns
        """)
        st.write("This page explores the distribution of the features throughout the dataset. We can see various stats of the data below the graph:")
        # Select column for distribution
        column = st.selectbox('Select a column to visualize', df_engineered.select_dtypes(include=['float64', 'int64']).columns)
            
        # Distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_engineered[column], kde=True, ax=ax)
        plt.title(f'Distribution of {column}')
        st.pyplot(fig)
            
        # Basic statistics for the selected column
        st.write(df_engineered[column].describe())


    # In the Seasonal Sales Analysis section:
    elif page == 'Seasonal Sales Analysis':
        st.header('Seasonal Sales Analysis')
        st.write("""
        Analyze store sales performance across different seasons. Select a season to see how different stores perform during that time period.
        """)
        st.info("""
        Key Features:
        • Compare store performance across seasons
        • Interactive seasonal sales visualizations
        • Top-performing stores by season
        • Season-specific sales patterns
        • Store-level seasonal comparison
        """)
        # Season selector
        selected_season = st.selectbox('Select Season:', ['Spring', 'Summer', 'Autumn', 'Winter'])
        
        # Filter data for selected season
        seasonal_data = df_engineered[df_engineered['Season'] == selected_season]
        
        # Calculate total sales per store for the selected season
        seasonal_store_sales = seasonal_data.groupby('Store')['Weekly_Sales'].sum().reset_index()
        
        # Create bar plot using plotly express
        fig_seasonal = px.bar(
            seasonal_store_sales,
            x='Store',
            y='Weekly_Sales',
            title=f'Store Sales During {selected_season}',
            labels={'Weekly_Sales': 'Total Sales ($)', 'Store': 'Store Number'}       
        )
        
        # Display the plot
        st.plotly_chart(fig_seasonal, use_container_width=True)
         
        # Get top 5 stores for selected season
        top_5_seasonal = seasonal_data.groupby('Store')['Weekly_Sales'].sum().reset_index()
        top_5_seasonal = top_5_seasonal.nlargest(5, 'Weekly_Sales')

        # Display detailed breakdown of top 5 stores
        st.subheader(f'Top 5 Performing Stores in {selected_season}')
    
        # Calculate average weekly sales for more meaningful comparison
        store_metrics = pd.DataFrame({
            'Store': top_5_seasonal['Store'],
            'Total Sales': top_5_seasonal['Weekly_Sales'],
            'Average Weekly Sales': top_5_seasonal['Weekly_Sales'] / seasonal_data.groupby('Store').size()[top_5_seasonal['Store']].values
        })
        
         # Create three columns for each store's metrics
        cols = st.columns(5)
        for idx, (_, store_data) in enumerate(store_metrics.iterrows()):
            with cols[idx]:
                # Format total sales in millions
                total_sales_m = store_data['Total Sales'] / 1_000_000
                # Format weekly sales in thousands
                weekly_sales_k = store_data['Average Weekly Sales'] / 1_000
                
                st.metric(
                    f"Store {int(store_data['Store'])}",
                    f"${total_sales_m:.2f}M",
                    f"${weekly_sales_k:.1f}K/week"
                )
        # Additional insights
        st.subheader('Seasonal Performance Insights')
        
        # Calculate and display season-specific metrics
        avg_seasonal_sales = seasonal_data['Weekly_Sales'].mean()
        max_seasonal_sales = seasonal_data['Weekly_Sales'].max()
        total_seasonal_sales = seasonal_data['Weekly_Sales'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Weekly Sales", f"${avg_seasonal_sales:,.2f}")
        with col2:
            st.metric("Maximum Weekly Sales", f"${max_seasonal_sales:,.2f}")
        with col3:
            st.metric("Total Seasonal Sales", f"${total_seasonal_sales:,.2f}")

        # Season-specific insights
        st.write(f"""
        **Key Observations for {selected_season}:**
        - The top performing store generated ${top_5_seasonal['Weekly_Sales'].max():,.2f} in total sales
        - The average sales among top 5 stores was ${top_5_seasonal['Weekly_Sales'].mean():,.2f}
        - The difference between the highest and lowest performing stores among the top 5 is ${(top_5_seasonal['Weekly_Sales'].max() - top_5_seasonal['Weekly_Sales'].min()):,.2f}
        """)
        
        
    elif page == 'Correlation Analysis':
        st.header('Correlation Analysis')
        
        # Select numeric columns for correlation
        numeric_cols = df_engineered.select_dtypes(include=['float64', 'int64']).columns
            
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df_engineered[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax, cbar=True)
        plt.title('Correlation Heatmap of Numeric Features')
        st.pyplot(fig)

        # Summary of findings
        st.write("""
        Strong Positive Correlations: 
        - Weekly_Sales and Sales_Log have a very strong positive correlation (0.96), which is expected since Sales_Log is derived from Weekly_Sales
        - This indicates the log transformation preserved the sales patterns while potentially normalizing the distribution""")
        st.write("""
        Notable Negative Correlations:
        - Store and Weekly_Sales show a moderate negative correlation (-0.34), suggesting that higher store numbers tend to have slightly lower sales
        - CPI and Unemployment have a moderate negative correlation (-0.3), indicating that as unemployment increases, the Consumer Price Index tends to decrease
        - Store and Sales_Log also show a similar negative correlation (-0.29) as Store and Weekly_Sales""")
        st.write("""
        Weak or No Correlations:
        - Holiday_Flag shows very weak correlations with most variables (all correlations < 0.16)
        - Temperature has weak correlations with all variables (all correlations ≤ 0.18)
        - Fuel_Price shows weak correlations with all other variables (all correlations ≤ 0.17)
        Economic Indicators:
        - The economic variables (CPI, Unemployment, Fuel_Price) show relatively weak correlations with sales
        - This suggests that short-term sales might be more influenced by other factors not captured in these economic indicators""")
                
        st.write("""This analysis suggests that while store location has some impact on sales, most of the external factors (temperature, holidays, economic indicators) have limited direct linear relationships with sales performance.
        """)

    elif page == 'Sales Forecasting':
        st.info("""
        Use Facebook's Prophet model to forecast future sales trends. This advanced forecasting tool accounts for multiple seasonal patterns and trends.
        
        Key Features:
        • Interactive forecast visualization
        • Adjustable forecast parameters
        • Trend and seasonality decomposition
        • Model performance metrics
        • Downloadable forecast results
        """)
        sales_forecasting_page()
    elif page == 'Summary':
            st.subheader("Forecasting Models Comparison")
            
            st.write("""
            This page provides a comprehensive comparison of the three forecasting models used in our analysis: 
            Prophet, ARIMA, and SARIMA. Understanding their differences helps in choosing the right model for 
            specific forecasting needs.
            """)

            # Model Overview
            st.subheader("Model Characteristics")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                model_select = st.radio(
                    "Select Model to Learn More:",
                    ["Prophet", "ARIMA", "SARIMA"]
                )
            
            with col2:
                if model_select == "Prophet":
                    st.write("""
                    **Facebook Prophet**
                    - Designed for business time series with strong seasonal patterns
                    - Handles missing data and outliers well
                    - Automatically detects trend changes
                    - Incorporates holiday effects
                    - Best for: Long-term forecasting with multiple seasonal patterns
                    """)
                elif model_select == "ARIMA":
                    st.write("""
                    **ARIMA (AutoRegressive Integrated Moving Average)**
                    - Focuses on linear relationships in time series
                    - Good for stationary or differenced stationary data
                    - Works well with short-term forecasting
                    - Doesn't handle seasonality directly
                    - Best for: Short-term forecasting of stable trends
                    """)
                else:  # SARIMA
                    st.write("""
                    **SARIMA (Seasonal ARIMA)**
                    - Extends ARIMA to handle seasonal patterns
                    - Combines trend and seasonal components
                    - More complex but more flexible than ARIMA
                    - Can capture multiple types of patterns
                    - Best for: Data with clear seasonal patterns
                    """)

            # Performance Comparison
            st.subheader("Performance Metrics Comparison")
            
            # Create a metrics comparison table
            metrics_df = pd.DataFrame({
                'Model': ['Prophet', 'ARIMA', 'SARIMA'],
                'Strengths': [
                    '• Handles multiple seasonality\n• Robust to missing data\n• Built-in trend changepoint detection',
                    '• Simple and interpretable\n• Good for short-term forecasts\n• Works well with stable trends',
                    '• Captures seasonal patterns\n• Flexible modeling options\n• Good for complex patterns'
                ],
                'Limitations': [
                    '• Can be computationally intensive\n• May overfit with limited data\n• Less transparent than simpler models',
                    '• Cannot handle multiple seasonality\n• Sensitive to non-stationarity\n• Limited with complex patterns',
                    '• More parameters to tune\n• Can be unstable with short series\n• Computationally intensive'
                ],
                'Best Use Cases': [
                    '• Long-term forecasting\n• Multiple seasonal patterns\n• Irregular data points',
                    '• Short-term forecasting\n• Stable, consistent trends\n• Quick preliminary analysis',
                    '• Clear seasonal patterns\n• Medium-term forecasting\n• Regular seasonal variations'
                ]
            })
            
            
            # Add interpretation of metrics
            st.write("""
            **Performance Analysis:**

            1. **Prophet Model Performance**
            - Shows the lowest MAE at $2.06M, indicating the best average prediction accuracy
            - Has the lowest MSE and RMSE, suggesting it handles outliers well
            - Demonstrates the most consistent performance across all metrics

            2. **ARIMA Model Performance**
            - Shows intermediate performance among the three models
            - Simpler model structure but still provides reasonable accuracy
            - Good for short-term forecasting needs

            3. **SARIMA Model Performance**
            - Higher MAE ($3.51M) suggests less accurate predictions on average
            - Significantly higher RMSE indicates larger prediction errors
            - Despite incorporating seasonality, doesn't outperform simpler models in this case

            **Key Takeaways:**
            - Prophet demonstrates superior performance across all metrics
            - The simpler ARIMA model performs better than the more complex SARIMA
            - The difference in MAE between best and worst performers is about $1.45M, which is significant for business planning
            """)
            st.table(metrics_df)

            # Recommendations
            st.subheader("Recommendations")
            st.write("""
            For Walmart sales forecasting:

            1. **Short-term Forecasting (1-4 weeks)**
            - ARIMA or SARIMA models are sufficient
            - Faster computation and good accuracy
            - Easier to update and maintain

            2. **Medium-term Forecasting (1-3 months)**
            - SARIMA or Prophet recommended
            - Better handling of seasonal patterns
            - More robust to changing trends

            3. **Long-term Forecasting (3+ months)**
            - Prophet is the preferred choice
            - Better handles multiple seasonality
            - More robust to long-term trend changes

            4. **Considerations for Implementation**
            - Regular model retraining is recommended
            - Monitor forecast accuracy over time
            - Consider ensemble approaches for critical decisions
            """)

            # Future Improvements
            st.subheader("Future Improvements")
            st.write("""
            To enhance forecasting accuracy:

            1. **Data Enrichment**
            - Include more external factors (e.g., local events, promotions)
            - Gather competitor data if possible
            - Incorporate more granular weather data

            2. **Model Enhancements**
            - Experiment with hybrid models
            - Implement ensemble methods
            - Add automated parameter tuning

            3. **Monitoring and Maintenance**
            - Set up automated accuracy tracking
            - Implement regular model retraining
            - Develop alert systems for forecast deviations
            """)



    elif page == 'ARIMA Forecasting':
        st.info("""
        Implement ARIMA (AutoRegressive Integrated Moving Average) models for time series forecasting. Perfect for capturing linear time series patterns.
        
        Key Features:
        • Configurable ARIMA parameters
        • Residual analysis and diagnostics
        • Model performance evaluation
        • Interactive forecast plots
        • Comparative analysis with actual values
        """)
        arima_forecasting_page()

    #elif page == 'SARIMA Forecasting':  
    #    st.info("""
    #    Use SARIMA (Seasonal ARIMA) models to capture both trend and seasonal patterns in the sales data. Ideal for data with strong seasonal components.
    #    
    #    Key Features:
    #    • Seasonal component modeling
    #    • Advanced parameter configuration
    #    • Multiple performance metrics
    #   • Animated forecast visualization
    #    • Detailed model diagnostics
    #    """)
    #    sarima_forecasting_page()

    elif page == 'Store Timeline':
        st.info("""
        Visualize the evolution of store sales over time with an interactive animated timeline. Track performance changes and identify trends.
        
        Key Features:
        • Animated sales timeline visualization
        • Store-specific performance metrics
        • Monthly trend analysis
        • Comparative store performance
        • Detailed statistical summaries
        """)
        store_timeline_page()
    

def amazon_sales_app():
    # Clear any existing Streamlit state
    st.empty()
    # Run the Amazon sales prediction app
    # Load data
    @st.cache_data
    def load_large_amazon_data():
        url = "https://media.githubusercontent.com/media/Diparna/Sales_Forecasting/refs/heads/main/Amazon%20Sales%20FY2020-21.csv" 
        df = pd.read_csv(url)
        
        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    
        return df

    def check_value_consistency(df):
        """Check consistency between price, quantity, value, and discounts"""
        # Create a copy of dataframe to avoid modifying original
        df_check = df.copy()
        
        # Calculate expected values
        df_check['calculated_value'] = df_check['price'] * df_check['qty_ordered']
        df_check['price_difference'] = df_check['calculated_value'] - df_check['value']
        
        # Calculate total after discount
        df_check['calculated_total'] = df_check['value'] - df_check['discount_amount']
        df_check['total_difference'] = df_check['calculated_total'] - df_check['total']
        
        consistency_check = {
            'Incorrect Value Calculations': len(df_check[abs(df_check['price_difference']) > 0.01]),
            'Incorrect Total Calculations': len(df_check[abs(df_check['total_difference']) > 0.01]),
            'Orders with Discounts': len(df_check[df_check['discount_amount'] > 0]),
            'Average Discount Amount': df_check['discount_amount'].mean()
        }
        
        return consistency_check, df_check

    def check_date_consistency(df):
        """Check consistency between order_date, year, and month"""
        # Create a copy of dataframe to avoid modifying original
        df_check = df.copy()
        
        # Extract year and month from order_date
        df_check['extracted_year'] = df_check['order_date'].dt.year
        df_check['extracted_month'] = df_check['order_date'].dt.month
        
        date_check = {
            'Year Mismatches': (df_check['extracted_year'] != df_check['year']).sum(),
            'Years in Data': sorted(df_check['year'].unique()),
            'Months with Orders': df_check.groupby(['year', 'month']).size().to_dict()
        }
        
        return date_check, df_check

    def perform_ida(df):
        """Perform comprehensive Initial Data Assessment"""
        # Basic Information
        basic_info = {
            'Total Records': len(df),
            'Total Columns': len(df.columns),
            'Unique Orders': df['order_id'].nunique(),
            'Date Range': f"{df['order_date'].min()} to {df['order_date'].max()}",
            'Total Items': df['qty_ordered'].sum()
        }
        
        # Missing Values Analysis
        missing_data = df.isnull().sum()
        missing_percentages = (missing_data / len(df) * 100).round(2)
        
        # Data Types
        data_types = df.dtypes
        
        # Order Analysis
        order_analysis = {
            'Avg Items per Order': df.groupby('order_id')['qty_ordered'].sum().mean(),
            'Max Items per Order': df.groupby('order_id')['qty_ordered'].sum().max(),
            'Unique Categories': df['category'].nunique(),
            'Unique Payment Methods': df['payment_method'].nunique(),
            'Unique Regions': df['Region'].nunique()
        }
        
        # Perform value consistency check
        value_check, df_value = check_value_consistency(df)
        
        # Perform date consistency check
        date_check, df_date = check_date_consistency(df)
        
        return {
            'basic_info': basic_info,
            'missing_data': missing_data,
            'missing_percentages': missing_percentages,
            'data_types': data_types,
            'order_analysis': order_analysis,
            'value_check': value_check,
            'date_check': date_check
        }, df_value, df_date


    def prepare_enhanced_features(df):
        """Prepare features including customer demographics and order details"""
        # Create a copy to avoid modifying original data
        df_features = df.copy()
        
        # Ensure order_date is datetime
        df_features['order_date'] = pd.to_datetime(df_features['order_date'])
        
        # Create time-based features first
        df_features['month'] = df_features['order_date'].dt.month
        df_features['day'] = df_features['order_date'].dt.day
        df_features['day_of_week'] = df_features['order_date'].dt.dayofweek
        df_features['quarter'] = df_features['order_date'].dt.quarter
        
        # Customer demographics - with error handling for age
        if 'age' in df_features.columns:
            df_features['age_group'] = pd.cut(df_features['age'], 
                                            bins=[0, 25, 35, 50, 65, 100],
                                            labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        else:
            df_features['age'] = 0  # default value if age is missing
            df_features['age_group'] = '0'
        
        # Region and payment method encoding with error handling
        label_encoders = {}
        categorical_cols = ['Region', 'payment_method', 'category', 'Gender']
        for col in categorical_cols:
            if col in df_features.columns:
                label_encoders[col] = LabelEncoder()
                df_features[f'{col}_encoded'] = label_encoders[col].fit_transform(df_features[col])
            else:
                df_features[f'{col}_encoded'] = 0  # default value if column is missing
        
        # Create daily aggregations with enhanced features
        agg_dict = {
            'total': 'sum',
            'qty_ordered': 'sum',
            'order_id': 'nunique'
        }
        
        # Add optional aggregations if columns exist
        if 'cust_id' in df_features.columns:
            agg_dict['cust_id'] = 'nunique'
        if 'discount_amount' in df_features.columns:
            agg_dict['discount_amount'] = 'mean'
        if 'age' in df_features.columns:
            agg_dict['age'] = 'mean'
        
        # Add encoded columns to aggregation
        for col in categorical_cols:
            if f'{col}_encoded' in df_features.columns:
                agg_dict[f'{col}_encoded'] = 'mean'
        
        daily_aggs = df_features.groupby('order_date').agg(agg_dict).reset_index()
        
        # Add rolling features
        windows = [7, 14, 30]
        for window in windows:
            daily_aggs[f'rolling_{window}d_sales'] = daily_aggs['total'].rolling(window=window, min_periods=1).mean()
            if 'cust_id' in agg_dict:
                daily_aggs[f'rolling_{window}d_customers'] = daily_aggs['cust_id'].rolling(window=window, min_periods=1).mean()
            daily_aggs[f'rolling_{window}d_qty'] = daily_aggs['qty_ordered'].rolling(window=window, min_periods=1).mean()
        
        # Add lag features
        lags = [1, 7, 14]
        for lag in lags:
            daily_aggs[f'sales_lag_{lag}'] = daily_aggs['total'].shift(lag)
            if 'cust_id' in agg_dict:
                daily_aggs[f'customers_lag_{lag}'] = daily_aggs['cust_id'].shift(lag)
            daily_aggs[f'qty_lag_{lag}'] = daily_aggs['qty_ordered'].shift(lag)
        
        # Calculate day-over-day changes
        daily_aggs['sales_change'] = daily_aggs['total'].pct_change()
        if 'cust_id' in agg_dict:
            daily_aggs['customer_change'] = daily_aggs['cust_id'].pct_change()
        
        # Fill NaN values with backward fill then forward fill
        daily_aggs = daily_aggs.fillna(method='bfill').fillna(method='ffill')
        
        return daily_aggs, label_encoders

    def train_enhanced_model(daily_aggs, forecast_days=30):
        """Train an enhanced sales prediction model"""
        # Define base features that we know exist
        base_features = ['month', 'day', 'day_of_week', 'quarter',
                        'qty_ordered',
                        'rolling_7d_sales', 'rolling_14d_sales', 'rolling_30d_sales',
                        'rolling_7d_qty', 'rolling_14d_qty', 'rolling_30d_qty',
                        'sales_lag_1', 'sales_lag_7', 'sales_lag_14',
                        'qty_lag_1', 'qty_lag_7', 'qty_lag_14',
                        'sales_change']
        
        # Add optional features if they exist
        features = []
        for feature in base_features:
            if feature in daily_aggs.columns:
                features.append(feature)
        
        # Add encoded categorical features if they exist
        categorical_cols = ['Region', 'payment_method', 'category', 'Gender']
        for col in categorical_cols:
            if f'{col}_encoded' in daily_aggs.columns:
                features.append(f'{col}_encoded')
        
        X = daily_aggs[features]
        y = daily_aggs['total']
        
        # Split data with a validation set
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, scaler, metrics, feature_importance, y_test, y_pred

    def analyze_predictions(y_test, y_pred, feature_importance, daily_aggs):
        """Create comprehensive prediction analysis"""
        # Actual vs Predicted plot
        comparison_df = pd.DataFrame({
            'Date': daily_aggs.iloc[-len(y_test):]['order_date'],
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        fig_timeline = px.line(comparison_df, x='Date', y=['Actual', 'Predicted'],
                            title='Sales Prediction Timeline',
                            labels={'value': 'Sales ($)', 'variable': 'Type'})
        
        # Scatter plot of Actual vs Predicted
        fig_scatter = px.scatter(comparison_df, x='Actual', y='Predicted',
                            title='Actual vs Predicted Sales',
                            labels={'Actual': 'Actual Sales ($)',
                                    'Predicted': 'Predicted Sales ($)'})
        fig_scatter.add_trace(px.line(x=[y_test.min(), y_test.max()],
                                    y=[y_test.min(), y_test.max()]).data[0])
        
        # Feature importance plot
        fig_importance = px.bar(feature_importance.head(15), x='importance', y='feature',
                            title='Top 15 Most Important Features',
                            labels={'importance': 'Importance Score',
                                    'feature': 'Feature'})
        
        return fig_timeline, fig_scatter, fig_importance

    def prepare_sarima_data(df):
        """Prepare data for SARIMA model"""
        # Convert to datetime if needed
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Create daily sales series
        daily_sales = df.groupby('order_date')['total'].sum().reset_index()
        daily_sales.set_index('order_date', inplace=True)
        
        # Ensure the index is continuous
        idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
        daily_sales = daily_sales.reindex(idx, fill_value=0)
        
        return daily_sales

    def analyze_time_series(daily_sales):
        """Analyze time series components"""
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(daily_sales['total'], period=7)
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=4, cols=1,
                        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        
        # Add traces
        fig.add_trace(go.Scatter(x=daily_sales.index, y=daily_sales['total'],
                                mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_sales.index, y=decomposition.trend,
                                mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_sales.index, y=decomposition.seasonal,
                                mode='lines', name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=daily_sales.index, y=decomposition.resid,
                                mode='lines', name='Residual'), row=4, col=1)
        
        fig.update_layout(height=900, title_text="Time Series Decomposition")
        return fig, decomposition


    def train_sarima_model(daily_sales, test_size=30):
        """Train SARIMA model with flexible forecast period"""
        # Split data into train and test
        train = daily_sales[:-test_size]
        test = daily_sales[-test_size:]
        
        # Fit SARIMA model
        model = SARIMAX(train['total'],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 7))
        
        results = model.fit(disp=False)
        
        return results, train, test

    def generate_forecast(results, test_size):
        """Generate forecast for specified number of days"""
        # Make predictions
        forecast = results.get_forecast(steps=test_size)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Calculate metrics
        test_data = daily_sales[-test_size:]
        if len(test_data) > len(forecast_mean):
            test_data = test_data[:len(forecast_mean)]
        elif len(test_data) < len(forecast_mean):
            forecast_mean = forecast_mean[:len(test_data)]
            forecast_ci = forecast_ci[:len(test_data)]
        
        mse = ((test_data['total'] - forecast_mean) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(test_data['total'] - forecast_mean).mean()
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse
        }
        
        return forecast_mean, forecast_ci, test_data, metrics

    def create_sarima_plots(daily_sales, forecast_mean, forecast_ci, test):
        """Create visualization plots for SARIMA predictions"""
        # Create the main prediction plot
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(x=daily_sales.index, y=daily_sales['total'],
                                mode='lines', name='Historical Sales',
                                line=dict(color='blue')))
        
        # Add forecasted values
        fig.add_trace(go.Scatter(x=test.index, y=forecast_mean,
                                mode='lines', name='Forecast',
                                line=dict(color='red')))
        
        # Add confidence intervals - fixing the attribute error
        fig.add_trace(go.Scatter(x=test.index.union(test.index[::-1]),
                                y=pd.concat([forecast_ci.iloc[:, 1], forecast_ci.iloc[:, 0][::-1]]),
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% Confidence Interval'))
        
        fig.update_layout(title='SARIMA Sales Forecast',
                        xaxis_title='Date',
                        yaxis_title='Sales',
                        showlegend=True)
        
        # Create actual vs predicted plot for test period
        scatter_fig = px.scatter(x=test['total'], y=forecast_mean,
                                labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
                                title='Actual vs Predicted Sales')
        
        # Add 45-degree line
        scatter_fig.add_trace(
            go.Scatter(x=[test['total'].min(), test['total'].max()],
                    y=[test['total'].min(), test['total'].max()],
                    mode='lines', name='Perfect Prediction',
                    line=dict(color='red', dash='dash'))
        )
        
        return fig, scatter_fig

    def format_metrics(metrics):
        """Format metrics for display"""
        return {k: f"${v/1000:,.1f}K" for k, v in metrics.items()}

    # Main app
    st.title('Large Amazon Dataset - Sales Prediction')

    # Load data
    with st.spinner('Loading data...'):
        df = load_large_amazon_data()
        st.success('Data loaded successfully!')

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Choose a page', 
        ['Sales Prediction',
        'IDA',
        'Order Analysis', 
        'Product Performance',
        'Financial Analysis',
        'Regional Analysis'
        ])

    if page == 'IDA':

        st.markdown("""
            This page provides a comprehensive overview of the Amazon sales dataset, analyzing data quality, 
            patterns, and basic statistics. The analysis helps understand the data's structure and identify 
            potential areas for deeper investigation.
            
            ### What You're Seeing:
            1. **Basic Information**: Key metrics about the dataset including total records, date range, and unique orders.
            2. **Missing Values Analysis**: Identifies any gaps in the data that might need attention.
            3. **Value Consistency Analysis**: Checks for discrepancies in price calculations and discounts.
            4. **Order Patterns**: Shows daily order volumes and distribution of items per order.
            5. **Regional Distribution**: Displays order distribution across different regions.
            """)

        # Perform IDA
        ida_results, df_value, df_date = perform_ida(df)

        # Display Results
        st.header('1. Basic Information')
        for key, value in ida_results['basic_info'].items():
            st.metric(key, value)

        st.header('2. Missing Values Analysis')
        missing_df = pd.DataFrame({
            'Missing Count': ida_results['missing_data'],
            'Missing Percentage': ida_results['missing_percentages']
        })
        st.write(missing_df[missing_df['Missing Count'] > 0])

        st.header('3. Value Consistency Analysis')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Incorrect Values", ida_results['value_check']['Incorrect Value Calculations'])
        with col2:
            st.metric("Incorrect Totals", ida_results['value_check']['Incorrect Total Calculations'])
        with col3:
            st.metric("Orders with Discounts", ida_results['value_check']['Orders with Discounts'])
        with col4:
            st.metric("Avg Discount", f"${ida_results['value_check']['Average Discount Amount']:.2f}")

        # Show some examples of inconsistencies if they exist
        if ida_results['value_check']['Incorrect Value Calculations'] > 0:
            st.subheader("Sample Value Inconsistencies")
            inconsistent_values = df_value[abs(df_value['price_difference']) > 0.01].head()
            st.write(inconsistent_values[['order_id', 'price', 'qty_ordered', 'value', 'calculated_value', 'price_difference']])

        st.header('4. Order Patterns')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Orders per Day')
            daily_orders = df.groupby('order_date')['order_id'].nunique()
            fig = px.line(x=daily_orders.index, y=daily_orders.values,
                        title='Daily Order Volume',
                        labels={'x': 'Date', 'y': 'Number of Orders'})
            st.plotly_chart(fig)

        with col2:
            st.subheader('Multiple Items per Order')
            items_per_order = df.groupby('order_id')['item_id'].count().value_counts()
            fig = px.bar(x=items_per_order.index, y=items_per_order.values,
                        title='Distribution of Items per Order',
                        labels={'x': 'Items in Order', 'y': 'Number of Orders'})
            st.plotly_chart(fig)

        st.header('5. Regional Distribution')
        region_counts = df['Region'].value_counts()
        fig = px.bar(x=region_counts.index, y=region_counts.values,
                    title='Orders by Region',
                    labels={'x': 'Region', 'y': 'Number of Orders'})
        st.plotly_chart(fig)

        # Sample Data
        st.header('6. Sample Data Preview')
        st.write("First few rows of the dataset:")
        st.write(df.head())

    elif page == 'Order Analysis':
        st.header('Order Analysis')
        
        st.markdown("""
            Analyze ordering patterns and trends over time. This page helps understand customer purchasing 
            behavior and identify peak sales periods.
            
            ### Key Features:
            - **Time Period Selection**: Filter data for specific date ranges
            - **Daily Order Volume**: Track how order volumes change over time
            - **Order Status Distribution**: See the breakdown of order statuses
            
            ### How to Use:
            1. Use the date selectors to choose your analysis period
            2. Observe daily patterns in the order volume chart
            3. Review the distribution of order statuses in the pie chart
            """)
        
        # Time period selector
        st.subheader('Select Time Period')
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', df['order_date'].min())
        with col2:
            end_date = st.date_input('End Date', df['order_date'].max())
        
        # Filter data by date range
        mask = (df['order_date'].dt.date >= start_date) & (df['order_date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        
        # Daily order volume
        st.subheader('Daily Order Volume')
        daily_orders = filtered_df.groupby('order_date')['order_id'].nunique()
        fig = px.line(x=daily_orders.index, y=daily_orders.values,
                    title='Daily Order Volume',
                    labels={'x': 'Date', 'y': 'Number of Orders'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Order status distribution
        st.subheader('Order Status Distribution')
        status_counts = filtered_df['status'].value_counts()
        fig = px.pie(values=status_counts.values, 
                    names=status_counts.index,
                    title='Distribution of Order Status')
        st.plotly_chart(fig, use_container_width=True)
        

    elif page == 'Product Performance':
        st.header('Product Performance')
        
        st.markdown("""
            Evaluate how different products and categories are performing in terms of sales and quantity sold.
            
            ### Understanding the Visualizations:
            1. **Top Selling Products**: 
            - Bar chart shows the best-performing products by quantity
            - Longer bars indicate higher sales volumes
            
            2. **Category Performance**: 
            - Pie chart shows revenue distribution across categories
            - Larger segments represent categories with higher revenue
            
            ### Using the Category Filter:
            - Select 'All' to view overall performance
            - Choose specific categories to drill down into their performance
            """)

        # Category selection
        selected_category = st.selectbox('Select Category:', 
                                    options=['All'] + list(df['category'].unique()))
        
        if selected_category != 'All':
            category_df = df[df['category'] == selected_category]
        else:
            category_df = df
        
        # Top selling products
        st.subheader('Top Selling Products')
        product_sales = category_df.groupby('sku')['qty_ordered'].sum().sort_values(ascending=False)
        
        fig = px.bar(x=product_sales.head(10).index, 
                    y=product_sales.head(10).values,
                    title='Top 10 Products by Quantity Sold',
                    labels={'x': 'SKU', 'y': 'Quantity Sold'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Category performance
        st.subheader('Category Performance')
        category_revenue = category_df.groupby('category')['total'].sum().sort_values(ascending=False)
        
        fig = px.pie(values=category_revenue.values,
                    names=category_revenue.index,
                    title='Revenue Distribution by Category')
        st.plotly_chart(fig, use_container_width=True)

    elif page == 'Financial Analysis':
        st.header('Financial Analysis')
        
        st.markdown("""
            Deep dive into the financial aspects of sales, including revenue trends, payment patterns, 
            and discount analysis.
            
            ### Dashboard Components:
            1. **Monthly Revenue Trend**:
            - Shows how revenue changes month over month
            - Helps identify seasonal patterns and growth trends
            
            2. **Payment Method Analysis**:
            - Distribution of different payment methods used
            - Insights into customer payment preferences
            
            """)
        # Monthly revenue trend
        st.subheader('Monthly Revenue Trend')
        monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['total'].sum()
        
        fig = px.line(x=monthly_revenue.index.astype(str), 
                    y=monthly_revenue.values,
                    title='Monthly Revenue',
                    labels={'x': 'Month', 'y': 'Revenue'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Payment method analysis
        st.subheader('Payment Method Analysis')
        payment_counts = df['payment_method'].value_counts()
        
        fig = px.pie(values=payment_counts.values,
                    names=payment_counts.index,
                    title='Distribution of Payment Methods')
        st.plotly_chart(fig, use_container_width=True)
        
        

    elif page == 'Regional Analysis':
        st.header('Regional Analysis')

        st.markdown("""
            Explore sales patterns across different geographic regions and understand regional preferences.
            
            ### Analysis Features:
            1. **Regional Sales Overview**:
            - Compare total sales across different regions
            - Identify strongest and weakest performing regions
            
            2. **Payment Preferences by Region**:
            - Understand how payment method preferences vary by region
            - Helps in regional payment strategy planning
            
            ### Using the Regional Filter:
            - Select specific regions to focus analysis
            - Compare patterns between different regions
            """)
        
        # Region selection
        selected_region = st.selectbox('Select Region:', 
                                    options=['All'] + list(df['Region'].unique()))
        
        if selected_region != 'All':
            region_df = df[df['Region'] == selected_region]
        else:
            region_df = df
        
        # Regional sales analysis
        st.subheader('Regional Sales Analysis')
        regional_sales = region_df.groupby('Region')['total'].sum().sort_values(ascending=False)
        
        fig = px.bar(x=regional_sales.index,
                    y=regional_sales.values,
                    title='Sales by Region',
                    labels={'x': 'Region', 'y': 'Total Sales'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Payment methods by region
        st.subheader('Payment Methods by Region')
        payment_by_region = pd.crosstab(region_df['Region'], region_df['payment_method'])
        
        fig = px.bar(payment_by_region,
                    title='Payment Methods Distribution by Region',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    elif page == 'Sales Prediction':
        st.header('Sales Prediction Analysis')

        st.markdown("""
            Forecast future sales using two different prediction models: Random Forest and SARIMA. Each model 
            offers different insights into future sales patterns.
        
            ### Random Forest Prediction:
            - Uses multiple features to predict sales
            - Shows feature importance and prediction accuracy
            - Good for understanding which factors influence sales
            
            ### SARIMA Prediction:
            - Specialized time series forecasting
            - Shows seasonal patterns and trends
            - Provides confidence intervals for predictions
            
            ### Using the Forecast Tools:
            1. Switch between tabs to compare different models
            2. Use the slider in SARIMA to adjust forecast period
            3. Review performance metrics to understand prediction accuracy
            """)
        
        st.write("We have displayed various statistics regarding the machine learning models and various things that come out of these analysis. Since this is for educational purposes we have included details and explanations!")
        
        # Create tabs for different models
        tab1, tab2 = st.tabs(["Random Forest Prediction", "SARIMA Prediction"])

        with tab1:
                with st.spinner('Preparing Random Forest prediction model...'):
                    daily_aggs, label_encoders = prepare_enhanced_features(df)
                    model, scaler, metrics, feature_importance, y_test, y_pred = train_enhanced_model(daily_aggs)
                    fig_timeline, fig_scatter, fig_importance = analyze_predictions(
                        y_test, y_pred, feature_importance, daily_aggs
                    )
                
                    # Display metrics in columns
                st.subheader('Model Performance Metrics')
                col1, col2, col3 = st.columns(3)
                with col1:
                    mae_k = metrics['MAE'] / 1000
                    st.metric('Mean Absolute Error', f"${mae_k:.1f}K")
                with col2:
                    rmse_k = metrics['RMSE'] / 1000
                    st.metric('Root Mean Square Error', f"${rmse_k:.1f}K")
                with col3:
                    st.metric('R² Score', f"{metrics['R2']:.3f}")
                
                # Display visualizations
                st.subheader('Sales Prediction Timeline')
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Create two columns for the remaining plots
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('Actual vs Predicted Sales')
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    st.subheader('Feature Importance')
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Add feature importance details in an expander
                with st.expander('Detailed Feature Importance Analysis'):
                    st.write('Top 10 Most Important Features:')
                    st.dataframe(feature_importance.head(10))
                # Add prediction explanation
                st.markdown("""
                ### Understanding the Predictions
            
                This model uses several types of features to make predictions:
                1. **Time-based patterns**: Day of week, month, and seasonal trends
                2. **Customer behavior**: Purchase patterns and demographics
                3. **Historical trends**: Rolling averages and lag features
                4. **Regional and payment patterns**: Geographic and payment method preferences
            
                The model's R² score indicates how well it explains the variance in sales. A score closer to 1.0 is better.
                """)

        with tab2:
            with st.spinner('Preparing SARIMA prediction model...'):
                # Prepare data for SARIMA
                daily_sales = prepare_sarima_data(df)
                
                # Add slider for prediction timeline
                forecast_days = st.slider(
                "Select number of days to forecast",
                min_value=7,
                max_value=180,
                value=30,
                step=7,
                help="Choose how many days ahead you want to predict"
                )

                # Analyze time series components
                decomp_fig, decomposition = analyze_time_series(daily_sales)
                
                # Train base SARIMA model
                results, train, test = train_sarima_model(daily_sales)
                
                # Generate forecast for selected period
                forecast_mean, forecast_ci, test_data, sarima_metrics = generate_forecast(
                results, forecast_days)


                # Create prediction plots
                forecast_fig, comparison_fig = create_sarima_plots(
                daily_sales, forecast_mean, forecast_ci, test_data)

                # Display SARIMA results
                st.subheader(f'SARIMA Model Performance ({forecast_days} Day Forecast)')
                formatted_metrics = format_metrics(sarima_metrics)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Mean Absolute Error', formatted_metrics['MAE'])
                with col2:
                    st.metric('Root Mean Square Error', formatted_metrics['RMSE'])
                with col3:
                    st.metric('Mean Square Error', formatted_metrics['MSE'])
                
                # Display forecast
                st.subheader(f'Sales Forecast (Next {forecast_days} Days)')
                st.plotly_chart(forecast_fig, use_container_width=True)

                # Display time series decomposition
                st.subheader('Time Series Decomposition')
                st.plotly_chart(decomp_fig, use_container_width=True)
                
                st.subheader('Understanding Time Series Decomposition')
                st.write("""
                Time series decomposition breaks down our sales data into four key components to better understand underlying patterns:

                1. **Original Data** (First Plot):
                - Shows the raw sales data over time
                - Includes all patterns, seasonality, and random fluctuations
                - Helps visualize the overall trend and complexity of the data

                2. **Trend** (Second Plot):
                - Represents the long-term progression of the sales
                - Removes seasonal and random variations
                - Shows whether sales are generally increasing, decreasing, or staying stable over time
                - Useful for understanding long-term business growth or decline

                3. **Seasonal Pattern** (Third Plot):
                - Shows recurring patterns at fixed intervals
                - In retail data, this often reflects:
                    * Weekly patterns (weekend vs. weekday sales)
                    * Monthly patterns (start/end of month variations)
                    * Holiday season impacts
                - Helps in planning inventory and staffing based on expected cycles

                4. **Residuals** (Fourth Plot):
                - Shows what remains after removing trend and seasonality
                - Represents random fluctuations or 'noise' in the data
                - Large residuals might indicate:
                    * Special events or promotions
                    * Unexpected sales spikes or drops
                    * Data quality issues

                **How to Use This Information:**
                - Use the trend to make long-term business decisions
                - Use seasonal patterns for short-term planning and scheduling
                - Monitor residuals to identify unusual events or potential issues
                - Combined analysis helps in making more accurate forecasts
                """)

def amazon_sentiment_app():
    # Clear any existing Streamlit state
    st.empty()
    # Run the Amazon sentiment analysis app
    # Load Amazon data
    @st.cache_data
    def load_amazon_data():
        url = "https://raw.githubusercontent.com/Diparna/Sales_Forecasting/refs/heads/main/amazon.csv" 
        df = pd.read_csv(url)
                
        # Convert price columns from string to numeric
        df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        
        # Convert rating_count to numeric
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
        # Convert rating to numeric if needed
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        return df

    # Initial Data Assessment
    def perform_ida(df):
        """Perform Initial Data Assessment"""
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percentages = (missing_data / total_rows * 100).round(2)
        
        # Data types
        data_types = df.dtypes
        
        # Duplicates
        duplicates = df.duplicated().sum()
        
        return {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'missing_data': missing_data,
            'missing_percentages': missing_percentages,
            'data_types': data_types,
            'duplicates': duplicates
        }

    # Transformation function
    @st.cache_data
    def transform_data(df):
        """Transform the dataset based on IDA findings"""
        df_transformed = df.copy()
        
        # Step 1: Drop img_link
        df_transformed = df_transformed.drop('img_link', axis=1)
        
        # Step 2: Create count features
        df_transformed['user_count'] = df_transformed['user_id'].str.count(',') + 1
        df_transformed['review_count'] = df_transformed['review_id'].str.count(',') + 1
        
        
        # Function to safely convert price strings to float
        def convert_price(price):
            if isinstance(price, str):
                return float(price.replace('₹', '').replace(',', ''))
            return float(price)
        
        # Function to safely convert discount percentage
        def convert_discount(discount):
            if isinstance(discount, str):
                return float(discount.replace('%', ''))
            return float(discount)
        
        # Convert price columns
        df['discounted_price'] = df['discounted_price'].apply(convert_price)
        df['actual_price'] = df['actual_price'].apply(convert_price)
        
        # Convert discount_percentage
        df['discount_percentage'] = df['discount_percentage'].apply(convert_discount)
        
        # Convert rating columns
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Step 3: Extract main category (before first '|')
        df_transformed['main_category'] = df_transformed['category'].str.split('|').str[0]
        # Get sub-category (second part between first and second '|')
        df_transformed['sub_category'] = df_transformed['category'].str.split('|').str[1]
        
    
        # Step 4: Drop original multiple-value columns
        final_df = df_transformed.drop(['user_id', 'user_name', 'review_id'], axis=1)
        
        return final_df, df_transformed

    def clean_text(text):
        """Clean and preprocess text data with better sentence preservation"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs and image references
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'https://m\.media-amazon\.com/images/\S+', '', text)
        
        # Replace comma with period to better separate sentences
        text = text.replace(',', '.')
        
        # Remove special characters but preserve sentence endings
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()

    def get_weighted_sentiment(text):
        """Get sentiment with more nuanced analysis"""
        sia = SentimentIntensityAnalyzer()
        
        if pd.isna(text) or text == "":
            return {
                'compound': 0,
                'pos': 0,
                'neu': 0,
                'neg': 0
            }
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return sia.polarity_scores(text)
        
        # Calculate sentiment for each sentence
        sentiments = [sia.polarity_scores(sentence) for sentence in sentences]
        
        # Weight recent sentences more heavily
        weights = np.linspace(0.5, 1.0, len(sentences))
        
        # Calculate weighted averages
        compound = np.average([s['compound'] for s in sentiments], weights=weights)
        pos = np.average([s['pos'] for s in sentiments], weights=weights)
        neg = np.average([s['neg'] for s in sentiments], weights=weights)
        neu = np.average([s['neu'] for s in sentiments], weights=weights)
        
        # Adjust compound score to be more discriminating
        compound = compound * 0.8  # Reduce overall positivity bias
        
        return {
            'compound': compound,
            'pos': pos,
            'neu': neu,
            'neg': neg
        }

    def analyze_review_sentiment(df):
        """Analyze sentiment with improved accuracy"""
        df_sentiment = df.copy()
        
        # Extract main category (before first '|')
        df_sentiment['main_category'] = df_sentiment['category'].str.split('|').str[0]
        # Get sub-category (second part between first and second '|')
        df_sentiment['sub_category'] = df_sentiment['category'].str.split('|').str[1]

        # Clean review title and content
        df_sentiment['cleaned_title'] = df_sentiment['review_title'].apply(clean_text)
        df_sentiment['cleaned_content'] = df_sentiment['review_content'].apply(clean_text)
        
        # Get weighted sentiments
        title_sentiments = df_sentiment['cleaned_title'].apply(get_weighted_sentiment)
        content_sentiments = df_sentiment['cleaned_content'].apply(get_weighted_sentiment)
        
        # Extract sentiment scores
        df_sentiment['title_compound'] = title_sentiments.apply(lambda x: x['compound'])
        df_sentiment['content_compound'] = content_sentiments.apply(lambda x: x['compound'])
        
        # Calculate overall sentiment with more weight to content
        df_sentiment['overall_sentiment'] = (
            df_sentiment['title_compound'] * 0.3 + 
            df_sentiment['content_compound'] * 0.7
        )
        
        # More granular sentiment labels
        df_sentiment['sentiment_label'] = df_sentiment['overall_sentiment'].apply(
            lambda x: 'Very Positive' if x >= 0.5 else
                    'Positive' if x >= 0.1 else
                    'Neutral' if x >= -0.1 else
                    'Negative' if x >= -0.5 else
                    'Very Negative'
        )
        
        # Add confidence score
        df_sentiment['sentiment_confidence'] = df_sentiment['overall_sentiment'].abs()
        
        return df_sentiment

    def display_sentiment_analysis(df_sentiment):
        """Create visualizations for sentiment analysis"""
        st.header("Review Sentiment Analysis")
        
        # Overall sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_sentiment['sentiment_label'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribution of Review Sentiments"
        )
        st.plotly_chart(fig)
        
        # Average sentiment by product category
        st.subheader("Average Sentiment by Category")
        category_sentiment = df_sentiment.groupby('main_category')['overall_sentiment'].mean().sort_values()
        fig = px.bar(
            x=category_sentiment.index,
            y=category_sentiment.values,
            title="Average Sentiment Score by Category"
        )
        st.plotly_chart(fig)
        
        # Sentiment vs Rating correlation
        st.subheader("Sentiment vs Rating Correlation")
        fig = px.scatter(
            df_sentiment,
            x='overall_sentiment',
            y='rating',
            title="Sentiment Score vs Product Rating",
            trendline="ols"
        )
        st.plotly_chart(fig)
        
        # Example reviews by sentiment
        st.subheader("Example Reviews by Sentiment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Most Positive Reviews")
            positive_reviews = df_sentiment.nlargest(5, 'overall_sentiment')[
                ['product_name', 'cleaned_title', 'cleaned_content', 'overall_sentiment']
            ]
            st.write(positive_reviews)
        
        with col2:
            st.write("Most Negative Reviews")
            negative_reviews = df_sentiment.nsmallest(5, 'overall_sentiment')[
                ['product_name', 'cleaned_title', 'cleaned_content', 'overall_sentiment']
            ]
            st.write(negative_reviews)

    def add_sentiment_analysis_page(df):
            df_sentiment = analyze_review_sentiment(df)
            display_sentiment_analysis(df_sentiment)
            return df_sentiment

    def generate_wordcloud(texts, title):
        """Generate and display wordcloud"""
        # Combine all texts
        text = ' '.join(texts)
        
        # Create and generate a word cloud image
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            min_font_size=10,
            max_font_size=150
        ).generate(text)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        return fig

    def create_sentiment_visualization(df_sentiment):
        """Create enhanced interactive sentiment visualization dashboard"""
        st.header("Sentiment Analysis Visualization")
        
        # Extract categories
        df_sentiment['main_category'] = df_sentiment['category'].str.split('|').str[0]
        df_sentiment['sub_category'] = df_sentiment['category'].str.split('|').str[1]
        
                
        # Create three columns for filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
        # Main category filter with "Select All" option
            main_categories = sorted(df_sentiment['main_category'].unique())
            main_categories = ['All Categories'] + main_categories
            selected_main_category = st.selectbox(
                "Select Main Category",
                options=main_categories,
                key='main_category_filter'
            )
        
        with filter_col2:
            # Sub category filter with "Select All" option
            if selected_main_category == 'All Categories':
                sub_categories = sorted(df_sentiment['sub_category'].unique())
            else:
                sub_categories = sorted(df_sentiment[
                    df_sentiment['main_category'] == selected_main_category
                ]['sub_category'].unique())
            
            sub_categories = ['All Subcategories'] + list(sub_categories)
            selected_sub_category = st.selectbox(
                "Select Sub Category",
                options=sub_categories,
                key='sub_category_filter'
            )

        with filter_col3:
            # Collapsible section for sentiment thresholds
            with st.expander("Adjust Sentiment Thresholds"):
                positive_threshold = st.slider(
                    "Positive Sentiment Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Reviews with sentiment score above this value are considered positive"
                )
                
                negative_threshold = st.slider(
                    "Negative Sentiment Threshold",
                    min_value=-1.0,
                    max_value=0.0,
                    value=-0.1,
                    step=0.1,
                    help="Reviews with sentiment score below this value are considered negative"
                )
        # Add a separator
        st.markdown("---")
        
        # Filter data based on selections
        if selected_main_category == 'All Categories':
            if selected_sub_category == 'All Subcategories':
                category_data = df_sentiment.copy()
            else:
                category_data = df_sentiment[
                    df_sentiment['sub_category'] == selected_sub_category
                ].copy()
        else:
            if selected_sub_category == 'All Subcategories':
                category_data = df_sentiment[
                    df_sentiment['main_category'] == selected_main_category
                ].copy()
            else:
                category_data = df_sentiment[
                    (df_sentiment['main_category'] == selected_main_category) &
                    (df_sentiment['sub_category'] == selected_sub_category)
                ].copy()
        
        # Apply dynamic sentiment thresholds
        category_data['sentiment_label'] = category_data['overall_sentiment'].apply(
            lambda x: 'Positive' if x >= positive_threshold else
                    'Negative' if x <= negative_threshold else
                    'Neutral'
        )
        
        # Show number of reviews and current filter selection
        if selected_main_category == 'All Categories':
            category_text = "All Categories"
        else:
            category_text = f"{selected_main_category}"
            if selected_sub_category != 'All Subcategories':
                category_text += f" - {selected_sub_category}"
                
        st.info(f"Analyzing {len(category_data)} reviews from {category_text}")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Sentiment Distribution", 
            "Detailed Analysis",
            "Rating Analysis",
            "Word Clouds"
        ])
        
        with tab1:
            # Previous sentiment distribution code remains the same
            col1, col2 = st.columns([2, 1])
            
            with col1:
                sentiment_dist = category_data['sentiment_label'].value_counts()
                
                fig_dist = px.pie(
                    values=sentiment_dist.values,
                    names=sentiment_dist.index,
                    title=f"Sentiment Distribution for {selected_main_category} - {selected_sub_category}",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.subheader("Summary Statistics")
                pos_pct = len(category_data[category_data['sentiment_label'] == 'Positive']) / len(category_data) * 100
                neu_pct = len(category_data[category_data['sentiment_label'] == 'Neutral']) / len(category_data) * 100
                neg_pct = len(category_data[category_data['sentiment_label'] == 'Negative']) / len(category_data) * 100
                
                st.metric("Positive Reviews", f"{pos_pct:.1f}%")
                st.metric("Neutral Reviews", f"{neu_pct:.1f}%")
                st.metric("Negative Reviews", f"{neg_pct:.1f}%")
                st.metric("Total Reviews", len(category_data))
        
        with tab2:
            # Previous detailed analysis code remains the same
            st.subheader("Sentiment Score Distribution")
            
            hist_data = np.histogram(category_data['overall_sentiment'], bins=30)
            max_height = max(hist_data[0])
            
            fig_hist = px.histogram(
                category_data,
                x='overall_sentiment',
                nbins=30,
                title=f"Distribution of Sentiment Scores",
                labels={'overall_sentiment': 'Sentiment Score'},
                color_discrete_sequence=['lightblue']
            )
            
            fig_hist.add_vline(x=positive_threshold, line_dash="dash", line_color="green")
            fig_hist.add_vline(x=negative_threshold, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab3:
            # Previous rating analysis code remains the same
            st.subheader("Sentiment vs Rating Analysis")
            
            if 'rating' in category_data.columns:
                fig_scatter = px.scatter(
                    category_data,
                    x='rating',
                    y='overall_sentiment',
                    title=f"Sentiment Score vs Rating",
                    trendline="ols",
                    labels={
                        'rating': 'Product Rating',
                        'overall_sentiment': 'Sentiment Score'
                    },
                    color='sentiment_label'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    correlation = category_data['rating'].corr(category_data['overall_sentiment'])
                    st.metric("Correlation: Rating vs Sentiment", f"{correlation:.2f}")
                
                with col2:
                    avg_sentiment = category_data['overall_sentiment'].mean()
                    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
        
        with tab4:
            st.subheader("Word Clouds by Sentiment")
            
            # Generate word clouds for each sentiment category
            sentiment_categories = ['Positive', 'Neutral', 'Negative']
            
            for sentiment in sentiment_categories:
                sentiment_texts = category_data[
                    category_data['sentiment_label'] == sentiment
                ]['cleaned_content'].dropna()
                
                if len(sentiment_texts) > 0:
                    st.subheader(f"{sentiment} Reviews Word Cloud")
                    wordcloud_fig = generate_wordcloud(
                        sentiment_texts,
                        f"Common Words in {sentiment} Reviews"
                    )
                    st.pyplot(wordcloud_fig)
                else:
                    st.write(f"No {sentiment.lower()} reviews available for word cloud generation")

    # Page content
    st.title('Amazon Products Analysis Dashboard')
    st.write("""
    This dashboard provides a comprehensive analysis of Amazon product data, including pricing, 
    ratings, reviews, and category insights.
    """)

    # Load data
    with st.spinner('Loading data...'):
        df = load_amazon_data()
        st.success('Data loaded successfully!')

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Choose a page', 
        ['Product Recommendations',
        'Initial Data Assessment',
        'Data Transformation', 
        'Product Overview',
        'Pricing Analysis',
        'Rating Analysis',
        'Sentiment Analysis'
        ])

    if page == 'Initial Data Assessment':
        st.header('Initial Data Assessment')
        
        # Perform IDA
        ida_results = perform_ida(df)
        
        # Display basic information
        st.subheader('Basic Information')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{ida_results['total_rows']:,}")
        with col2:
            st.metric("Total Features", f"{ida_results['total_columns']:,}")
        with col3:
            st.metric("Duplicate Records", f"{ida_results['duplicates']:,}")
        
        # Display missing values
        st.subheader('Missing Values Analysis')
        missing_df = pd.DataFrame({
            'Missing Count': ida_results['missing_data'],
            'Missing Percentage': ida_results['missing_percentages']
        })
        st.write(missing_df[missing_df['Missing Count'] > 0])
        
        # Data types
        st.subheader('Data Types')
        st.write(pd.DataFrame({'Data Type': ida_results['data_types']}))
        
        # Sample data
        st.subheader('Sample Data Preview')
        st.write(df.head())

    elif page == 'Data Transformation':
        st.header('Data Transformation Steps')
        
        st.write("""
        Based on our initial data assessment, we'll perform the following transformations:
        1. Remove unnecessary columns (img_link)
        2. Handle multiple values in user and review columns by creating count features
        3. Remove original multiple-value columns
        """)
        
        # Get transformed data
        final_df, df_transformed = transform_data(df)
        
        # Show original data structure
        st.subheader("Original Data Structure")
        st.write("Original columns:", df.columns.tolist())
        st.write("Sample of original data:")
        st.write(df.head(2))
        
        # Step 1: Drop img_link
        st.subheader("Step 1: Drop img_link column")
        st.write("Columns after dropping img_link:")
        st.write(df_transformed.columns.tolist())
        
        # Step 2: Create count features
        st.subheader("Step 2: Create count features")
        st.write("Example of how we handle multiple values:")
        example_df = pd.DataFrame({
            'product_name': df_transformed['product_name'].head(2),
            'Original user_id': df_transformed['user_id'].head(2),
            'Calculated user_count': df_transformed['user_count'].head(2),
            'Original review_id': df_transformed['review_id'].head(2),
            'Calculated review_count': df_transformed['review_count'].head(2)
        })
        st.write(example_df)
        
        # Step 3: Extract Main Category
        st.subheader("Step 3: Extract Main Category")
        st.write("Example of category extraction:")
        example_categories = pd.DataFrame({
            'Original Category': df_transformed['category'].head(),
            'Main Category': df_transformed['main_category'].head(),
            'Sub Category': df_transformed['sub_category'].head()
        })
        st.write(example_categories)

        # Step 4: Show final structure
        st.subheader("Step 3: Final Data Structure")
        st.write("Final columns:", final_df.columns.tolist())
        st.write("Sample of final transformed data:")
        st.write(final_df.head(2))
        
        # Transformation summary
        st.subheader("Transformation Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Columns", len(df.columns))
        with col2:
            st.metric("Final Columns", len(final_df.columns))
        with col3:
            st.metric("Columns Removed", len(df.columns) - len(final_df.columns))

    elif page == 'Product Overview':
        final_df, _ = transform_data(df)
        st.header('Product Overview')
        
        # Category distribution
        st.subheader('Product Categories Distribution')
        category_counts = final_df['main_category'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title='Number of Products by Category'
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="",
            xaxis={'tickangle': 45}
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Top products by rating and review count
        st.subheader('Top Rated Products (with significant reviews)')
        min_reviews = st.slider('Minimum number of reviews', 1, 1000, 100)
        
        top_products = final_df[final_df['rating_count'] >= min_reviews].nlargest(10, 'rating')
        
        fig = px.bar(
            top_products,
            x='product_name',
            y='rating',
            title=f'Top Rated Products (min {min_reviews} reviews)',
            hover_data=['rating_count']
        )

        fig.update_layout(
            xaxis_title="Product Name",  # Changed from product_name
            yaxis_title="Rating",
            xaxis={'tickangle': 45},
            xaxis_ticktext=[name[:50] + '...' if len(name) > 50 else name 
                        for name in top_products['product_name']],
            xaxis_tickvals=list(range(len(top_products)))
        )

        st.plotly_chart(fig, use_container_width=True)

        # Get unique categories
        categories = sorted(final_df['main_category'].unique())
        selected_category = st.selectbox('Select Product Category:', categories)
        
        # Filter data for selected category
        category_df = final_df[final_df['main_category'] == selected_category]
        
        # Get top 10 products in selected category based on ratings
        top_products = category_df.nlargest(10, 'rating')
        
        # Create bar chart for top products
        fig = px.bar(
            top_products,
            x='product_name',
            y='rating',
            title=f'Top 10 Rated Products in {selected_category}',
            hover_data=['actual_price', 'discounted_price', 'discount_percentage', 'rating_count']
        )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Product Name",  # Changed from product_name
            yaxis_title="Rating",
            xaxis={'tickangle': 45},
            xaxis_ticktext=[name[:40] + '...' if len(name) > 40 else name 
                        for name in top_products['product_name']],
            xaxis_tickvals=list(range(len(top_products)))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(top_products)

    elif page == 'Pricing Analysis':
        final_df, _ = transform_data(df)
        st.header('Pricing Analysis')

        # Add Category Selector
        st.subheader('Category-wise Product Analysis')

        categories = sorted(final_df['main_category'].unique())
        selected_category = st.selectbox('Select Product Category:', categories)
        category_df = final_df[final_df['main_category'] == selected_category]
        # Get top 10 products in selected category based on ratings
        top_products = category_df.nlargest(10, 'rating')
        
        # Display detailed information in a table
        st.subheader('Detailed Information for Top Products')
        display_cols = ['product_name', 'rating', 'rating_count', 
                    'actual_price', 'discounted_price', 'discount_percentage']
        st.write(top_products[display_cols])

        # Discount analysis
        st.subheader('Discount Analysis')
        
        fig = px.histogram(
            final_df,
            x='discount_percentage',
            title='Distribution of Discount Percentages',
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary section using columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Common Discounts")
            # Get mode of discount percentage
            common_discount = final_df['discount_percentage'].mode().iloc[0]
            st.write(f"Most frequent discount: {common_discount}%")
            
            # Count of products with this discount
            count_common = len(final_df[final_df['discount_percentage'] == common_discount])
            st.write(f"Number of products with {common_discount}% discount: {count_common}")
            
            # Show some products with this discount
            st.write("Sample products with this discount:")
            sample_products = final_df[final_df['discount_percentage'] == common_discount][['product_name', 'actual_price', 'discounted_price']].head(3)
            st.write(sample_products)
        
        with col2:
            st.subheader("Highest Discounted Top-Rated Products")
            # Get products with high ratings and high discounts
            # Filter and sort
            top_discounted = (final_df[final_df['rating'] >= 4.0]
                        .sort_values('discount_percentage', ascending=False)
                        .head(5))
            
            for _, row in top_discounted.iterrows():
                st.write(f"""
                - **{row['product_name']}**
                - Rating: {row['rating']} ({row['rating_count']} reviews)
                - Discount: {row['discount_percentage']}%
                - Original Price: ₹{row['actual_price']:,.2f}
                - Discounted Price: ₹{row['discounted_price']:,.2f}
                """)

        # Price by category
        st.subheader('Price Range by Category')
        
        fig = px.box(
            final_df,
            x='main_category',
            y='actual_price',
            title='Price Distribution by Category'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == 'Rating Analysis':
        # Get transformed data
        try:
            final_df, _ = transform_data(df)
            
            st.header('Rating Analysis')
            
            # Create filters
            st.subheader('Filter Products')
            
            try:
                # Main Category filter
                main_categories = sorted(final_df['main_category'].unique())
                if not main_categories:
                    st.error("No categories found in the data. Please check the data transformation.")
                else:
                    selected_main_cat = st.selectbox(
                        'Select Main Category:',
                        options=main_categories,
                        key='main_cat'
                    )
                    
                    # Filter data by main category
                    filtered_df = final_df[final_df['main_category'] == selected_main_cat]
                    
                    if filtered_df.empty:
                        st.warning(f"No products found in the category: {selected_main_cat}")
                    else:
                        # Sub Category filter
                        sub_categories = sorted(filtered_df['sub_category'].unique())
                        if sub_categories:
                            selected_sub_cat = st.selectbox(
                                'Select Sub Category:',
                                options=sub_categories,
                                key='sub_cat'
                            )
                            
                            # Final filtered data
                            filtered_df = filtered_df[filtered_df['sub_category'] == selected_sub_cat]
                            
                            if filtered_df.empty:
                                st.warning(f"No products found in the sub-category: {selected_sub_cat}")
                            else:
                                # Rating distribution for filtered data
                                st.subheader(f'Rating Distribution for {selected_main_cat} - {selected_sub_cat}')
                                
                                fig = px.histogram(
                                    filtered_df,
                                    x='rating',
                                    title=f'Distribution of Ratings',
                                    nbins=20
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Additional statistics for the filtered category
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_rating = filtered_df['rating'].mean()
                                    st.metric("Average Rating", f"{avg_rating:.2f}")
                                
                                with col2:
                                    total_products = len(filtered_df)
                                    st.metric("Total Products", total_products)
                                
                                with col3:
                                    avg_reviews = filtered_df['rating_count'].mean()
                                    st.metric("Average Reviews", f"{avg_reviews:.0f}")
                                
                                # Top rated products in selected category
                                st.subheader('Top Rated Products in Selected Category')
                                top_rated = filtered_df.sort_values('rating', ascending=False).head(5)
                                
                                for _, row in top_rated.iterrows():
                                    st.write(f"""
                                    - **{row['product_name']}**
                                    - Rating: {row['rating']} ({row['rating_count']} reviews)
                                    - Price: ₹{row['actual_price']:,.2f}
                                    - Discount: {row['discount_percentage']}%
                                    """)
                        else:
                            st.warning(f"No sub-categories found for: {selected_main_cat}")
                            
            except Exception as e:
                st.error(f"An error occurred while loading the data: {str(e)}")
                st.error("Please check your data transformation steps and try again.")
            
        except Exception as e:
            st.error(f"Failed to transform the data: {str(e)}")


    elif page == 'Sentiment Analysis':
        df_sentiment = add_sentiment_analysis_page(df)
        create_sentiment_visualization(df_sentiment)
    
    elif page == "Product Recommendations":
        st.header('Category-Based Recommendations')
        st.write("According to the selected category, this page will show you recommendations! Our dataset is limited so there might be very few products in some categories. We apologize for the incovenience! The data keeps getting updated!")
        final_df, df_transformed = transform_data(df)
    
        # Category selection
        main_categories = sorted(df_transformed['main_category'].unique())
        selected_main_cat = st.selectbox(
            'Select Main Category:',
            options=main_categories
        )
        
        # Filter sub-categories based on main category
        sub_categories = sorted(df_transformed[
            df_transformed['main_category'] == selected_main_cat]['sub_category'].unique())
        selected_sub_cat = st.selectbox(
            'Select Sub-Category:',
            options=sub_categories
        )
        
        # Filter data for selected categories
        filtered_data = df_transformed[
            (df_transformed['main_category'] == selected_main_cat) & 
            (df_transformed['sub_category'] == selected_sub_cat)
        ]
            
        # Calculate metrics
        df_sentiment = analyze_review_sentiment(filtered_data)
        avg_sentiment = df_sentiment['overall_sentiment'].mean()
        avg_rating = df_sentiment['rating'].mean()
        positive_pct = (df_sentiment['sentiment_label'] == 'Positive').mean() * 100
        negative_pct = (df_sentiment['sentiment_label'] == 'Negative').mean() * 100
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        with col2:
            st.metric("Average Rating", f"{avg_rating:.1f}⭐")
        with col3:
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        with col4:
            st.metric("Negative Reviews", f"{negative_pct:.1f}%")
        
        # Generate recommendations
        st.subheader("Recommendations")
        
        # Overall sentiment status
        if avg_sentiment >= 0.5:
            st.success("""
            ### Strong Category Performance 📈
            
            **Key Strengths:**
            - High customer satisfaction
            - Strong positive sentiment
            - Reliable product performance
            
            **Recommended Actions:**
            1. Maintain current product quality standards
            2. Consider expanding product range
            3. Highlight positive customer experiences in marketing
            4. Monitor for consistent performance
            """)
        elif avg_sentiment <= -0.1:
            st.error("""
            ### Areas Needing Attention ⚠️
            
            **Key Concerns:**
            - Lower customer satisfaction
            - Presence of negative sentiment
            - Potential product issues
            
            **Recommended Actions:**
            1. Review common customer complaints
            2. Implement immediate improvement measures
            3. Enhance customer support
            4. Follow up with dissatisfied customers
            """)
        else:
            st.info("""
            ### Mixed Performance 📊
            
            **Current Status:**
            - Moderate customer satisfaction
            - Mixed customer feedback
            - Room for improvement
            
            **Recommended Actions:**
            1. Address specific customer concerns
            2. Enhance product features
            3. Improve customer communication
            4. Monitor customer feedback closely
            """)
        
        # Show top reviews
        st.subheader("Review Highlights")
        st.write("**Most Positive Reviews:**")
        top_positive = df_sentiment[
            df_sentiment['sentiment_label'] == 'Positive'
        ].nlargest(10, 'overall_sentiment')[['product_name','cleaned_title','rating' , 'overall_sentiment']]
        st.dataframe(top_positive)
    
        st.write("**Most Critical Reviews:**")
        top_negative = df_sentiment[
            df_sentiment['sentiment_label'] == 'Negative'
        ].nsmallest(10, 'overall_sentiment')[['product_name','cleaned_title','rating' ,'overall_sentiment']]
        st.dataframe(top_negative)

if __name__ == "__main__":
    main()
