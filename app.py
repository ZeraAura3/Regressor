import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🧠 Regression Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/regression-simulator',
        'Report a bug': 'https://github.com/your-repo/regression-simulator/issues',
        'About': "# Interactive Regression Simulator\nA comprehensive tool for regression analysis!"
    }
)

# Enhanced responsive CSS for better visual appeal and adaptive sizing
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding-top: clamp(1rem, 3vw, 2rem);
        padding-bottom: clamp(2rem, 4vw, 3rem);
        max-width: 100%;
        overflow-x: hidden;
    }
    
    /* Responsive Container */
    .main-container {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 clamp(0.5rem, 2vw, 1rem);
    }
    
    /* Main Header - Responsive */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(2rem, 6vw, 3.5rem);
        font-weight: 700;
        text-align: center;
        margin: clamp(1rem, 3vw, 2rem) 0 clamp(1.5rem, 4vw, 3rem) 0;
        padding: clamp(0.75rem, 2vw, 1rem);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        position: relative;
        word-wrap: break-word;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: clamp(60px, 15vw, 100px);
        height: clamp(3px, 0.5vw, 4px);
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Subtitle - Responsive */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(0.9rem, 2.5vw, 1.2rem);
        color: #6c757d;
        text-align: center;
        margin-bottom: clamp(1rem, 3vw, 2rem);
        font-weight: 300;
        line-height: 1.5;
    }
    
    /* Sub Headers - Responsive */
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(1.2rem, 3vw, 1.8rem);
        font-weight: 600;
        color: #2c3e50;
        margin: clamp(1rem, 3vw, 2rem) 0 clamp(0.75rem, 2vw, 1.5rem) 0;
        padding: clamp(0.6rem, 1.5vw, 0.8rem) clamp(0.8rem, 2vw, 1.2rem);
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: clamp(3px, 0.8vw, 5px) solid #667eea;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    /* Status Cards - Responsive */
    .status-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: clamp(1rem, 2.5vw, 1.5rem);
        margin: clamp(0.5rem, 1.5vw, 1rem) 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        word-wrap: break-word;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Metric Cards - Responsive */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: clamp(1rem, 2.5vw, 1.5rem);
        border-radius: 15px;
        margin: clamp(0.4rem, 1vw, 0.8rem) 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
        min-height: clamp(100px, 15vw, 130px);
        display: flex;
        flex-direction: column;
        justify-content: center;
        word-wrap: break-word;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: clamp(1.25rem, 4vw, 2rem);
        font-weight: 700;
        margin-bottom: 0.5rem;
        word-break: break-word;
    }
    
    .metric-label {
        font-size: clamp(0.7rem, 1.8vw, 0.9rem);
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        word-wrap: break-word;
    }
    
    /* Alert Boxes - Responsive */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        border-radius: 12px;
        padding: clamp(1rem, 2vw, 1.2rem) clamp(1rem, 2.5vw, 1.5rem);
        margin: clamp(1rem, 2vw, 1.5rem) 0;
        border-left: clamp(3px, 0.8vw, 5px) solid #28a745;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.2);
        font-family: 'Poppins', sans-serif;
        font-size: clamp(0.85rem, 2vw, 1rem);
        word-wrap: break-word;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: none;
        border-radius: 12px;
        padding: clamp(1rem, 2vw, 1.2rem) clamp(1rem, 2.5vw, 1.5rem);
        margin: clamp(1rem, 2vw, 1.5rem) 0;
        border-left: clamp(3px, 0.8vw, 5px) solid #dc3545;
        box-shadow: 0 3px 10px rgba(220, 53, 69, 0.2);
        font-family: 'Poppins', sans-serif;
        font-size: clamp(0.85rem, 2vw, 1rem);
        word-wrap: break-word;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: none;
        border-radius: 12px;
        padding: clamp(1rem, 2vw, 1.2rem) clamp(1rem, 2.5vw, 1.5rem);
        margin: clamp(1rem, 2vw, 1.5rem) 0;
        border-left: clamp(3px, 0.8vw, 5px) solid #17a2b8;
        box-shadow: 0 3px 10px rgba(23, 162, 184, 0.2);
        font-family: 'Poppins', sans-serif;
        font-size: clamp(0.85rem, 2vw, 1rem);
        word-wrap: break-word;
    }
    
    /* Feature Cards - Responsive */
    .feature-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: clamp(1rem, 2.5vw, 1.5rem);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: clamp(0.5rem, 1.5vw, 1rem) 0;
        min-height: clamp(120px, 20vw, 200px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        word-wrap: break-word;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }
    
    .feature-card h3 {
        font-size: clamp(1rem, 2.5vw, 1.25rem);
        margin-bottom: clamp(0.5rem, 1.5vw, 1rem);
    }
    
    /* Tab Content - Responsive */
    .tab-content {
        padding: clamp(1rem, 3vw, 2rem) clamp(0.5rem, 2vw, 1rem);
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        margin: 1rem 0;
        word-wrap: break-word;
    }
    
    /* Sidebar Styling - Responsive */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 2px solid #e9ecef;
        padding-top: clamp(0.5rem, 1.5vw, 1rem);
    }
    
    /* Tab Styling - Responsive */
    .stTabs [data-baseweb="tab-list"] {
        gap: clamp(4px, 1vw, 8px);
        margin-bottom: clamp(1rem, 2vw, 1.5rem);
        background: #f8f9fa;
        padding: clamp(4px, 1vw, 8px);
        border-radius: 15px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        overflow-x: auto;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        border-radius: 12px;
        background: transparent;
        border: none;
        padding: clamp(8px, 2vw, 12px) clamp(12px, 3vw, 20px);
        color: #6c757d;
        transition: all 0.3s ease;
        font-size: clamp(0.8rem, 2vw, 1rem);
        white-space: nowrap;
        min-width: max-content;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Styling - Responsive */
    .stButton > button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        border-radius: 12px;
        border: none;
        padding: clamp(0.5rem, 1.5vw, 0.6rem) clamp(1rem, 2.5vw, 1.5rem);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: clamp(0.8rem, 2vw, 0.9rem);
        width: 100%;
        min-height: 44px; /* Touch-friendly minimum */
        word-wrap: break-word;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
    }
    
    /* Form Elements - Responsive */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSlider > div > div > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        transition: border-color 0.3s ease;
        font-size: clamp(0.8rem, 2vw, 1rem);
        min-height: 44px; /* Touch-friendly */
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Data Frame Styling - Responsive */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: clamp(1rem, 2vw, 1.5rem) 0;
        font-size: clamp(0.7rem, 1.8vw, 0.9rem);
        width: 100%;
        overflow-x: auto;
    }
    
    .stDataFrame {
        font-size: clamp(0.7rem, 1.5vw, 0.9rem);
    }
    
    /* Spacing Classes - Responsive */
    .section-spacing {
        margin: clamp(1.5rem, 4vw, 2.5rem) 0;
        padding: clamp(0.5rem, 1.5vw, 1rem) 0;
    }
    
    .content-spacing {
        margin: clamp(1rem, 2.5vw, 1.5rem) 0;
    }
    
    .small-spacing {
        margin: clamp(0.5rem, 1.2vw, 0.8rem) 0;
    }
    
    /* Footer - Responsive */
    .footer-section {
        margin-top: clamp(2rem, 4vw, 3rem);
        padding: clamp(1.5rem, 3vw, 2rem) 0;
        border-top: 2px solid #e9ecef;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px 15px 0 0;
    }
    
    .footer-section h3 {
        font-size: clamp(1rem, 2.5vw, 1.25rem);
        margin-bottom: clamp(0.5rem, 1vw, 1rem);
    }
    
    .footer-card {
        background: white;
        padding: clamp(1rem, 2.5vw, 1.5rem);
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: clamp(0.25rem, 1vw, 0.5rem);
        transition: transform 0.3s ease;
        word-wrap: break-word;
    }
    
    .footer-card:hover {
        transform: translateY(-2px);
    }
    
    /* Welcome Section - Responsive */
    .welcome-container, .welcome-section {
        text-align: center;
        padding: clamp(2rem, 5vw, 4rem) clamp(1rem, 3vw, 2rem);
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 20px;
        margin: clamp(1rem, 3vw, 2rem) 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .welcome-title {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: clamp(1rem, 2vw, 1.5rem);
        word-wrap: break-word;
    }
    
    .welcome-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(1rem, 2.5vw, 1.2rem);
        color: #6c757d;
        margin-bottom: 1rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    
    .welcome-description {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(0.9rem, 2vw, 1rem);
        color: #868e96;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    .version-info {
        text-align: center;
        padding: clamp(0.75rem, 2vw, 1rem);
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        font-size: clamp(0.8rem, 2vw, 1rem);
        word-wrap: break-word;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: clamp(16px, 3vw, 20px);
        height: clamp(16px, 3vw, 20px);
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top: 3px solid #667eea;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive Plotly Charts */
    .js-plotly-plot {
        width: 100% !important;
        height: auto !important;
        min-height: clamp(250px, 40vw, 500px);
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .stColumns {
            flex-direction: column;
        }
        
        .metric-card {
            margin: 0.5rem 0;
            min-height: 80px;
            padding: 1rem;
        }
        
        .feature-card {
            min-height: 120px;
            margin: 0.75rem 0;
        }
        
        .main-header {
            padding: 1rem 0.5rem;
            margin: 1rem 0 1.5rem 0;
        }
        
        .tab-content {
            padding: 1rem 0.5rem;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: scroll;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }
        
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        
        .stSidebar {
            width: 100% !important;
        }
    }
    
    /* Tablet adjustments */
    @media (max-width: 1024px) and (min-width: 769px) {
        .metric-card {
            min-height: 100px;
            padding: 1.25rem;
        }
        
        .feature-card {
            min-height: 160px;
        }
        
        .main-header {
            font-size: 2.8rem;
        }
    }
    
    /* Large screen optimizations */
    @media (min-width: 1400px) {
        .main {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .metric-card {
            max-width: 250px;
            margin: 0.8rem auto;
        }
        
        .feature-card {
            max-width: 300px;
        }
    }
    
    /* High DPI / Retina displays */
    @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
        .main-header, .sub-header, .metric-card {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
    }
    
    /* Ensure text doesn't overflow */
    * {
        word-wrap: break-word;
        overflow-wrap: break-word;
        box-sizing: border-box;
    }
    
    /* Container queries support */
    .stContainer {
        container-type: inline-size;
    }
    
    /* Enhanced Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: clamp(8px, 1.5vw, 12px);
    }
    
    /* Loading States */
    .stSpinner {
        border-top-color: #667eea !important;
        width: clamp(24px, 4vw, 32px) !important;
        height: clamp(24px, 4vw, 32px) !important;
    }
    
    /* Focus indicators for accessibility */
    .stButton > button:focus,
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        outline: 2px solid #667eea;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

class RegressionSimulatorApp:
    def __init__(self):
        self.initialize_session_state()
    
    @st.cache_data
    def load_sample_data_cached(_self, dataset_type, n_samples=200):
        """Load sample datasets with caching for better performance"""
        np.random.seed(42)
        
        if dataset_type == "Housing Data":
            area = np.random.uniform(500, 5000, n_samples)
            bedrooms = np.random.randint(1, 6, n_samples)
            bathrooms = np.random.randint(1, 4, n_samples)
            age = np.random.uniform(0, 50, n_samples)
            
            price = 50000 + 100*area + 25000*bedrooms + 30000*bathrooms - 2000*age + np.random.normal(0, 50000, n_samples)
            
            return pd.DataFrame({
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'price': price
            }), ['area', 'bedrooms', 'bathrooms', 'age'], 'price'
            
        elif dataset_type == "Simple Linear":
            x = np.linspace(0, 10, n_samples)
            y = 2*x + 5 + np.random.normal(0, 2, n_samples)
            
            return pd.DataFrame({'x': x, 'y': y}), ['x'], 'y'
            
        elif dataset_type == "Nonlinear Data":
            x = np.linspace(-5, 5, n_samples)
            y = 2*x**2 - 3*x + 1 + np.random.normal(0, 5, n_samples)
            
            return pd.DataFrame({'x': x, 'y': y}), ['x'], 'y'
        
        elif dataset_type == "Boston Housing":
            # Simulate Boston housing data
            features = np.random.randn(n_samples, 5)
            feature_names = ['CRIM', 'RM', 'AGE', 'DIS', 'LSTAT']
            target = (10 + 2*features[:, 1] - 0.5*features[:, 2] + 
                     0.8*features[:, 3] - 1.2*features[:, 4] + 
                     np.random.normal(0, 3, n_samples))
            
            data = pd.DataFrame(features, columns=feature_names)
            data['MEDV'] = target
            
            return data, feature_names, 'MEDV'
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data': None,
            'X': None,
            'y': None,
            'feature_names': [],
            'target_name': "",
            'models': {},
            'results': {},
            'analysis_history': [],
            'app_version': '1.0.0'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_sample_data(self, dataset_type):
        """Load sample datasets"""
        try:
            data, feature_names, target_name = self.load_sample_data_cached(dataset_type)
            
            st.session_state.data = data
            st.session_state.feature_names = feature_names
            st.session_state.target_name = target_name
            st.session_state.X = data[feature_names].values
            st.session_state.y = data[target_name].values
            
            # Clear previous models when new data is loaded
            st.session_state.models = {}
            st.session_state.results = {}
            
            return True
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return False
    
    def train_model(self, model_type, **kwargs):
        """Train regression models with error handling"""
        if st.session_state.X is None:
            st.error("Please load data first!")
            return None, None
        
        try:
            # Split data
            test_size = kwargs.get('test_size', 0.2)
            random_state = kwargs.get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X, st.session_state.y, 
                test_size=test_size, random_state=random_state
            )
            
            # Initialize model based on type
            if model_type == "Linear":
                model = LinearRegression()
                model.fit(X_train, y_train)
                model_name = "Linear Regression"
                
                # Store model
                st.session_state.models[model_name] = model
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
            elif model_type == "Polynomial":
                degree = kwargs.get('degree', 2)
                poly = PolynomialFeatures(degree=degree, include_bias=True)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                model_name = f"Polynomial (degree={degree})"
                
                # Store both transformer and model
                st.session_state.models[model_name] = {'poly': poly, 'linear': model}
                
                # Make predictions
                y_train_pred = model.predict(X_train_poly)
                y_test_pred = model.predict(X_test_poly)
                
            elif model_type == "Ridge":
                alpha = kwargs.get('alpha', 1.0)
                model = Ridge(alpha=alpha, random_state=random_state)
                model.fit(X_train, y_train)
                model_name = f"Ridge (α={alpha})"
                
                # Store model
                st.session_state.models[model_name] = model
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
            elif model_type == "Lasso":
                alpha = kwargs.get('alpha', 1.0)
                model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
                model.fit(X_train, y_train)
                model_name = f"Lasso (α={alpha})"
                
                # Store model
                st.session_state.models[model_name] = model
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            
            else:
                st.error(f"Unknown model type: {model_type}")
                return None, None
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            
            # Calculate additional metrics
            train_mae = np.mean(np.abs(y_train - y_train_pred))
            test_mae = np.mean(np.abs(y_test - y_test_pred))
            
            # Store results
            results = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'training_time': time.time()
            }
            
            # Add model-specific information
            if hasattr(model, 'coef_'):
                results['coefficients'] = model.coef_
            if hasattr(model, 'intercept_'):
                results['intercept'] = model.intercept_
            if model_type == "Polynomial":
                results['degree'] = degree
            if model_type in ["Ridge", "Lasso"]:
                results['alpha'] = kwargs.get('alpha', 1.0)
            
            # Store results
            st.session_state.results[model_name] = results
            
            # Add to analysis history
            st.session_state.analysis_history.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'action': f'Trained {model_name}',
                'test_r2': test_r2
            })
            
            return model_name, results
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None
    
    def create_data_visualization(self):
        """Create data visualization plots"""
        if st.session_state.data is None:
            st.warning("Please load data first!")
            return
        
        data = st.session_state.data
        
        if len(st.session_state.feature_names) == 1:
            # Simple scatter plot
            fig = px.scatter(
                data, 
                x=st.session_state.feature_names[0], 
                y=st.session_state.target_name,
                title=f"{st.session_state.target_name} vs {st.session_state.feature_names[0]}",
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            return fig
            
        elif len(st.session_state.feature_names) == 2:
            # 3D scatter plot
            fig = px.scatter_3d(
                data,
                x=st.session_state.feature_names[0],
                y=st.session_state.feature_names[1],
                z=st.session_state.target_name,
                title=f"3D Visualization: {st.session_state.target_name} vs Features",
                template="plotly_white"
            )
            return fig
            
        else:
            # Correlation heatmap
            corr_matrix = data[st.session_state.feature_names + [st.session_state.target_name]].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Heatmap",
                template="plotly_white"
            )
            return fig
    
    def create_model_predictions_plot(self, model_name):
        """Create model predictions visualization"""
        if model_name not in st.session_state.results:
            st.error("Model not found!")
            return None
        
        results = st.session_state.results[model_name]
        
        if len(st.session_state.feature_names) == 1:
            # 1D regression line
            feature_name = st.session_state.feature_names[0]
            
            # Create prediction line
            x_range = np.linspace(
                st.session_state.data[feature_name].min(),
                st.session_state.data[feature_name].max(),
                100
            ).reshape(-1, 1)
            
            if 'Polynomial' in model_name:
                poly = st.session_state.models[model_name]['poly']
                linear = st.session_state.models[model_name]['linear']
                x_poly = poly.transform(x_range)
                y_pred_line = linear.predict(x_poly)
            else:
                model = st.session_state.models[model_name]
                y_pred_line = model.predict(x_range)
            
            fig = go.Figure()
            
            # Add training data
            fig.add_trace(go.Scatter(
                x=results['X_train'].flatten(),
                y=results['y_train'],
                mode='markers',
                name='Training Data',
                marker=dict(color='blue', size=8, opacity=0.7)
            ))
            
            # Add test data
            fig.add_trace(go.Scatter(
                x=results['X_test'].flatten(),
                y=results['y_test'],
                mode='markers',
                name='Test Data',
                marker=dict(color='green', size=8, opacity=0.7)
            ))
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_pred_line,
                mode='lines',
                name=f'{model_name} Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f"{model_name} Predictions",
                xaxis_title=feature_name,
                yaxis_title=st.session_state.target_name,
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Actual vs Predicted plot for multiple features
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training Data', 'Test Data')
            )
            
            # Training data
            fig.add_trace(
                go.Scatter(
                    x=results['y_train'],
                    y=results['y_train_pred'],
                    mode='markers',
                    name='Training',
                    marker=dict(color='blue', size=8, opacity=0.7)
                ),
                row=1, col=1
            )
            
            # Perfect prediction line (training)
            min_val = min(results['y_train'].min(), results['y_train_pred'].min())
            max_val = max(results['y_train'].max(), results['y_train_pred'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Test data
            fig.add_trace(
                go.Scatter(
                    x=results['y_test'],
                    y=results['y_test_pred'],
                    mode='markers',
                    name='Test',
                    marker=dict(color='green', size=8, opacity=0.7)
                ),
                row=1, col=2
            )
            
            # Perfect prediction line (test)
            min_val = min(results['y_test'].min(), results['y_test_pred'].min())
            max_val = max(results['y_test'].max(), results['y_test_pred'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Actual Values")
            fig.update_yaxes(title_text="Predicted Values")
            fig.update_layout(
                title=f"{model_name} - Actual vs Predicted",
                template="plotly_white"
            )
            
            return fig
    
    def create_model_comparison_plot(self):
        """Create model comparison visualization"""
        if not st.session_state.results:
            st.warning("No models trained yet!")
            return None
        
        model_names = list(st.session_state.results.keys())
        train_r2 = [results['train_r2'] for results in st.session_state.results.values()]
        test_r2 = [results['test_r2'] for results in st.session_state.results.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Training R²',
            x=model_names,
            y=train_r2,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Test R²',
            x=model_names,
            y=test_r2,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Model Comparison - R² Scores',
            xaxis_title='Model',
            yaxis_title='R² Score',
            barmode='group',
            template="plotly_white"
        )
        
        return fig
    
    def create_residual_analysis(self, model_name):
        """Create residual analysis plots"""
        if model_name not in st.session_state.results:
            st.error("Model not found!")
            return None
        
        results = st.session_state.results[model_name]
        train_residuals = results['y_train'] - results['y_train_pred']
        test_residuals = results['y_test'] - results['y_test_pred']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals vs Predicted (Training)',
                'Residuals vs Predicted (Test)',
                'Residual Distribution (Training)',
                'Residual Distribution (Test)'
            )
        )
        
        # Residuals vs Predicted (Training)
        fig.add_trace(
            go.Scatter(
                x=results['y_train_pred'],
                y=train_residuals,
                mode='markers',
                name='Training Residuals',
                marker=dict(color='blue', size=6, opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals vs Predicted (Test)
        fig.add_trace(
            go.Scatter(
                x=results['y_test_pred'],
                y=test_residuals,
                mode='markers',
                name='Test Residuals',
                marker=dict(color='green', size=6, opacity=0.7)
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # Histogram of residuals (Training)
        fig.add_trace(
            go.Histogram(
                x=train_residuals,
                name='Training Residuals',
                marker_color='blue',
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Histogram of residuals (Test)
        fig.add_trace(
            go.Histogram(
                x=test_residuals,
                name='Test Residuals',
                marker_color='green',
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Residual Analysis - {model_name}',
            template="plotly_white",
            showlegend=False
        )
        
        return fig

def main():
    # Initialize the app
    app = RegressionSimulatorApp()
    
    # Header with improved design
    st.markdown('<h1 class="main-header">🧠 Interactive Regression Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A comprehensive tool for regression analysis, model comparison, and visualization</p>', unsafe_allow_html=True)
    
    # Status indicator section with better spacing
    st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**🚀 Deploy-ready Streamlit application**")
        st.markdown("✨ *Production-optimized with enhanced UI/UX*")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">v1.0.0</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Version</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        if st.session_state.data is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.data.shape[0]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Samples</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">0</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Samples</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        if st.button("🔄 Reset App", help="Clear all data and models"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🎛️ Control Panel</h2>', unsafe_allow_html=True)
        
        # Data Management Section
        st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Management")
        
        # Data loading options with better spacing
        data_option = st.selectbox(
            "Choose data source:",
            ["Select...", "Sample Datasets", "Upload CSV", "Manual Entry"],
            help="Select how you want to load your data"
        )
        
        if data_option == "Sample Datasets":
            st.markdown('<div class="small-spacing">', unsafe_allow_html=True)
            dataset_type = st.selectbox(
                "Select sample dataset:",
                ["Housing Data", "Simple Linear", "Nonlinear Data", "Boston Housing"],
                help="Choose from built-in datasets for quick testing"
            )
            
            # Add sample size selector for performance
            n_samples = st.slider("Number of samples:", 50, 1000, 200, 50,
                                help="Adjust dataset size for performance")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Load Sample Data", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading data..."):
                    success = app.load_sample_data(dataset_type)
                    if success:
                        st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
        
        elif data_option == "Upload CSV":
            st.markdown('<div class="small-spacing">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv",
                                           help="Upload your own dataset (max 10MB)")
            
            if uploaded_file is not None:
                try:
                    # Add file size check for deployment
                    file_size = uploaded_file.size / (1024 * 1024)  # MB
                    if file_size > 10:  # 10MB limit
                        st.markdown('<div class="error-box">❌ File size too large! Please upload a file smaller than 10MB.</div>', unsafe_allow_html=True)
                    else:
                        data = pd.read_csv(uploaded_file)
                        
                        with st.expander("📋 Data Preview", expanded=True):
                            st.dataframe(data.head(), use_container_width=True)
                            st.info(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
                        
                        # Column selection with better layout
                        columns = data.columns.tolist()
                        target_col = st.selectbox("🎯 Select target variable:", columns,
                                                help="Choose the variable you want to predict")
                        feature_cols = st.multiselect("📊 Select feature variables:", 
                                                    [col for col in columns if col != target_col],
                                                    help="Choose the input variables for prediction")
                        
                        if st.button("📤 Load CSV Data", type="primary", use_container_width=True) and feature_cols:
                            # Validate data
                            if len(feature_cols) == 0:
                                st.markdown('<div class="error-box">❌ Please select at least one feature!</div>', unsafe_allow_html=True)
                            elif data[feature_cols + [target_col]].isnull().sum().sum() > 0:
                                st.markdown('<div class="info-box">⚠️ Data contains missing values. They will be dropped.</div>', unsafe_allow_html=True)
                                data = data[feature_cols + [target_col]].dropna()
                            
                            st.session_state.data = data
                            st.session_state.feature_names = feature_cols
                            st.session_state.target_name = target_col
                            st.session_state.X = data[feature_cols].values
                            st.session_state.y = data[target_col].values
                            
                            # Clear previous models
                            st.session_state.models = {}
                            st.session_state.results = {}
                            
                            st.markdown('<div class="success-box">✅ CSV data loaded successfully!</div>', unsafe_allow_html=True)
                            st.rerun()
                            
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Error loading CSV: {e}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Training Section
        if st.session_state.data is not None:
            st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
            st.markdown("### 🤖 Model Training")
            
            model_type = st.selectbox(
                "Select model type:",
                ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression"],
                help="Choose the regression algorithm to train"
            )
            
            # Advanced options expander with better styling
            with st.expander("⚙️ Advanced Training Options"):
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05,
                                        help="Proportion of data for testing")
                with col2:
                    random_state = st.number_input("Random state:", 0, 1000, 42,
                                                 help="Seed for reproducible results")
            
            # Model-specific parameters with enhanced warnings
            model_params = {'test_size': test_size, 'random_state': random_state}
            
            if model_type == "Polynomial Regression":
                degree = st.slider("Polynomial degree:", 2, 10, 2,
                                 help="Higher degrees may lead to overfitting")
                model_params['degree'] = degree
                
                # Enhanced warning for high degrees
                if degree > 5:
                    st.markdown('<div class="info-box">⚠️ High polynomial degrees may cause overfitting! Consider using regularization.</div>', unsafe_allow_html=True)
                    
            elif model_type in ["Ridge Regression", "Lasso Regression"]:
                alpha = st.slider("Regularization strength (α):", 0.001, 10.0, 1.0, step=0.001, format="%.3f",
                                help="Higher values = more regularization")
                model_params['alpha'] = alpha
                
                if alpha < 0.01:
                    st.markdown('<div class="info-box">💡 Very low α values may lead to overfitting</div>', unsafe_allow_html=True)
                elif alpha > 5:
                    st.markdown('<div class="info-box">💡 High α values may lead to underfitting</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                model_key = model_type.split()[0]  # Linear, Polynomial, Ridge, Lasso
                
                with st.spinner(f"🔄 Training {model_type}..."):
                    model_name, results = app.train_model(model_key, **model_params)
                    
                    if model_name and results:
                        st.markdown('<div class="success-box">✅ Model trained successfully!</div>', unsafe_allow_html=True)
                        
                        # Show quick metrics with improved layout
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Test R²", f"{results['test_r2']:.4f}", 
                                    delta=f"{results['test_r2'] - results['train_r2']:.4f}")
                        with metric_col2:
                            st.metric("Test RMSE", f"{results['test_rmse']:.2f}")
                        
                        time.sleep(1)
                        st.rerun()
            
            # Quick model comparison with enhanced styling
            if len(st.session_state.results) > 0:
                st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                st.markdown("### 📊 Quick Model Overview")
                
                best_model = max(st.session_state.results.items(), key=lambda x: x[1]['test_r2'])
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"🏆 **Best Model:** {best_model[0]}")
                st.markdown(f"📈 **R² Score:** {best_model[1]['test_r2']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Total Models Trained:** {len(st.session_state.results)}")
                
                # Progress bar for model performance
                progress_value = min(best_model[1]['test_r2'], 1.0)
                st.progress(progress_value)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Deployment info footer with enhanced styling
        st.markdown('<div class="footer-section">', unsafe_allow_html=True)
        st.markdown("### 🚀 Deployment Ready")
        
        deployment_options = ["Streamlit Cloud", "Heroku", "Docker", "AWS/GCP"]
        for i, option in enumerate(deployment_options):
            if i % 2 == 0:
                st.markdown(f"✅ {option}")
            else:
                st.markdown(f"✅ {option}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with enhanced spacing
    if st.session_state.data is not None:
        # Enhanced tabs with consistent spacing
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Explorer", 
            "🎯 Model Predictions", 
            "📈 Model Comparison", 
            "🔍 Residual Analysis", 
            "� Export Center"
        ])
        
        with tab1:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📊 Data Explorer</h3>', unsafe_allow_html=True)
            
            # Enhanced data overview section
            st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
            
            # Data summary cards with better layout
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{st.session_state.data.shape[0]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Samples</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{len(st.session_state.feature_names)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Features</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{st.session_state.target_name}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Target</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                if hasattr(st.session_state, 'y'):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{st.session_state.y.mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Target Mean</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data preview section with enhanced styling
            st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
            
            # Interactive data display controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("#### 📋 Dataset Preview")
            with col2:
                show_rows = st.selectbox("Rows to show:", [5, 10, 20, 50], index=1, key="preview_rows")
            with col3:
                show_stats = st.checkbox("Show statistics", value=True, key="show_stats")
            
            # Enhanced data display
            if show_stats:
                with st.expander("📊 Statistical Summary", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Descriptive Statistics:**")
                        st.dataframe(st.session_state.data.describe(), use_container_width=True)
                    with col2:
                        st.markdown("**Data Types & Missing Values:**")
                        info_df = pd.DataFrame({
                            'Data Type': st.session_state.data.dtypes,
                            'Missing Values': st.session_state.data.isnull().sum(),
                            'Missing %': (st.session_state.data.isnull().sum() / len(st.session_state.data) * 100).round(2)
                        })
                        st.dataframe(info_df, use_container_width=True)
            
            # Enhanced data preview with styling
            st.markdown("**Sample Data:**")
            st.dataframe(st.session_state.data.head(show_rows), use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced visualization section
            st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📊 Data Visualization")
            with col2:
                plot_type = st.selectbox("Plot type:", 
                                       ["Scatter Plot", "Distribution", "Correlation Matrix", "Pair Plot"],
                                       key="data_plot_type")
            
            # Enhanced visualization with better loading states
            with st.spinner("🎨 Generating visualization..."):
                try:
                    fig = app.create_data_visualization()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Error creating visualization: {e}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">🎯 Model Performance Dashboard</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
                
                # Enhanced model comparison table
                st.markdown("#### 📊 Model Performance Comparison")
                
                comparison_data = []
                for model_name, results in st.session_state.results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Train R²': f"{results['train_r2']:.4f}",
                        'Test R²': f"{results['test_r2']:.4f}",
                        'Train MSE': f"{results['train_mse']:.4f}",
                        'Test MSE': f"{results['test_mse']:.4f}",
                        'Overfitting': f"{abs(results['train_r2'] - results['test_r2']):.4f}"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced model comparison chart
                st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                st.markdown("#### 📈 Visual Model Comparison")
                
                with st.spinner("🎨 Creating comparison chart..."):
                    fig = app.create_model_comparison_plot()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Best model highlight
                st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                best_model_name = max(st.session_state.results.items(), key=lambda x: x[1]['test_r2'])[0]
                best_results = st.session_state.results[best_model_name]
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"### 🏆 Best Performing Model: {best_model_name}")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Test R²", f"{best_results['test_r2']:.4f}")
                with metric_col2:
                    st.metric("Test MSE", f"{best_results['test_mse']:.4f}")
                with metric_col3:
                    st.metric("Train R²", f"{best_results['train_r2']:.4f}")
                with metric_col4:
                    overfitting = abs(best_results['train_r2'] - best_results['test_r2'])
                    st.metric("Overfitting", f"{overfitting:.4f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("🤖 **No models trained yet!**")
                st.markdown("Train some models using the sidebar controls to see performance metrics here.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📈 Interactive Model Visualizations</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
                
                # Enhanced model selection
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_model = st.selectbox(
                        "Select model to visualize:",
                        list(st.session_state.results.keys()),
                        help="Choose which trained model to analyze"
                    )
                with col2:
                    show_confidence = st.checkbox("Show confidence intervals", 
                                                value=True, 
                                                help="Display prediction uncertainty bands")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if selected_model:
                    # Enhanced model predictions
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Model Predictions vs Actual Values")
                    
                    with st.spinner("🎨 Creating prediction visualization..."):
                        try:
                            fig = app.create_model_predictions_plot(selected_model)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                            else:
                                st.markdown('<div class="info-box">📊 Prediction plot requires univariate data or will show actual vs predicted scatter plot for multivariate data.</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error creating prediction plot: {e}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced interactive prediction section
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 🔮 Interactive Prediction Tool")
                    st.markdown("Enter feature values to get a real-time prediction:")
                    
                    prediction_values = []
                    
                    # Create responsive columns based on number of features
                    n_features = len(st.session_state.feature_names)
                    if n_features <= 3:
                        cols = st.columns(n_features)
                    else:
                        cols = st.columns(3)
                    
                    for i, feature in enumerate(st.session_state.feature_names):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            min_val = float(st.session_state.data[feature].min())
                            max_val = float(st.session_state.data[feature].max())
                            default_val = float(st.session_state.data[feature].mean())
                            
                            value = st.number_input(
                                f"{feature}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"pred_{feature}_{selected_model}",
                                help=f"Range: {min_val:.2f} to {max_val:.2f}"
                            )
                            prediction_values.append(value)
                    
                    # Enhanced prediction button and result display
                    if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
                        try:
                            features = np.array(prediction_values).reshape(1, -1)
                            
                            if 'Polynomial' in selected_model:
                                poly = st.session_state.models[selected_model]['poly']
                                linear = st.session_state.models[selected_model]['linear']
                                features_poly = poly.transform(features)
                                prediction = linear.predict(features_poly)[0]
                            else:
                                model = st.session_state.models[selected_model]
                                prediction = model.predict(features)[0]
                            
                            # Display prediction with enhanced styling
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"### 🎯 Prediction Result")
                            st.markdown(f"**{st.session_state.target_name}: {prediction:.3f}**")
                            
                            # Add confidence information if available
                            model_results = st.session_state.results[selected_model]
                            rmse = model_results['test_rmse']
                            st.markdown(f"*Typical prediction error (RMSE): ±{rmse:.3f}*")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Prediction error: {e}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("🤖 **No models available for visualization!**")
                st.markdown("Train some models using the sidebar controls to see interactive visualizations here.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">🔍 Advanced Residual Analysis</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
                
                # Enhanced analysis controls
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_model = st.selectbox(
                        "Select model for residual analysis:",
                        list(st.session_state.results.keys()),
                        key="analysis_model",
                        help="Choose which model to analyze for residuals and diagnostic plots"
                    )
                with col2:
                    analysis_type = st.selectbox(
                        "Analysis type:",
                        ["Residual Plots", "Cross-Validation", "Feature Importance"],
                        help="Select the type of analysis to perform"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if analysis_type == "Residual Plots":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Residual Analysis Plots")
                    
                    with st.spinner("🎨 Generating residual analysis..."):
                        try:
                            fig = app.create_residual_analysis(selected_model)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error creating residual analysis: {e}</div>', unsafe_allow_html=True)
                    
                    # Enhanced residual statistics
                    st.markdown('<div class="small-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 📈 Residual Statistics")
                    
                    try:
                        results = st.session_state.results[selected_model]
                        train_residuals = results['y_train'] - results['y_train_pred']
                        test_residuals = results['y_test'] - results['y_test_pred']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-value">{np.mean(train_residuals):.4f}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="metric-label">Train Residual Mean</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-value">{np.std(train_residuals):.4f}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="metric-label">Train Residual Std</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-value">{np.mean(test_residuals):.4f}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="metric-label">Test Residual Mean</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-value">{np.std(test_residuals):.4f}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="metric-label">Test Residual Std</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Diagnostic insights
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**🔍 Diagnostic Insights:**")
                        if abs(np.mean(train_residuals)) < 0.01:
                            st.markdown("✅ Training residuals are well-centered around zero")
                        else:
                            st.markdown("⚠️ Training residuals show potential bias")
                        
                        if abs(np.mean(test_residuals)) < 0.01:
                            st.markdown("✅ Test residuals are well-centered around zero")
                        else:
                            st.markdown("⚠️ Test residuals show potential bias")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error calculating residual statistics: {e}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                elif analysis_type == "Cross-Validation":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 🔄 Cross-Validation Analysis")
                    
                    n_folds = st.slider("Number of folds:", 3, 10, 5, help="Number of cross-validation folds")
                    
                    if st.button("🚀 Run Cross-Validation", type="primary"):
                        with st.spinner("🔄 Running cross-validation analysis..."):
                            try:
                                if 'Linear' in selected_model:
                                    from sklearn.linear_model import LinearRegression
                                    model = LinearRegression()
                                    cv_scores = cross_val_score(model, st.session_state.X, st.session_state.y, cv=n_folds, scoring='r2')
                                elif 'Polynomial' in selected_model:
                                    degree = st.session_state.results[selected_model].get('degree', 2)
                                    poly = PolynomialFeatures(degree=degree)
                                    X_poly = poly.fit_transform(st.session_state.X)
                                    model = LinearRegression()
                                    cv_scores = cross_val_score(model, X_poly, st.session_state.y, cv=n_folds, scoring='r2')
                                elif 'Ridge' in selected_model:
                                    from sklearn.linear_model import Ridge
                                    alpha = st.session_state.results[selected_model].get('alpha', 1.0)
                                    model = Ridge(alpha=alpha)
                                    cv_scores = cross_val_score(model, st.session_state.X, st.session_state.y, cv=n_folds, scoring='r2')
                                elif 'Lasso' in selected_model:
                                    from sklearn.linear_model import Lasso
                                    alpha = st.session_state.results[selected_model].get('alpha', 1.0)
                                    model = Lasso(alpha=alpha)
                                    cv_scores = cross_val_score(model, st.session_state.X, st.session_state.y, cv=n_folds, scoring='r2')
                                
                                # Enhanced CV results display
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="metric-value">{np.mean(cv_scores):.4f}</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">Mean CV Score</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col2:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="metric-value">{np.std(cv_scores):.4f}</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">Std CV Score</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col3:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="metric-value">{np.min(cv_scores):.4f}</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">Min CV Score</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col4:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="metric-value">{np.max(cv_scores):.4f}</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">Max CV Score</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Enhanced CV scores plot
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                                    y=cv_scores,
                                    marker_color='lightblue',
                                    text=[f"{score:.3f}" for score in cv_scores],
                                    textposition='auto'
                                ))
                                
                                # Add mean line
                                fig.add_hline(y=np.mean(cv_scores), line_dash="dash", line_color="red",
                                            annotation_text=f"Mean: {np.mean(cv_scores):.3f}")
                                
                                fig.update_layout(
                                    title=f"{n_folds}-Fold Cross-Validation Scores for {selected_model}",
                                    xaxis_title="Fold",
                                    yaxis_title="R² Score",
                                    template="plotly_white",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                                
                                # CV interpretation
                                cv_std = np.std(cv_scores)
                                if cv_std < 0.05:
                                    st.markdown('<div class="success-box">✅ Low variance in CV scores indicates stable model performance</div>', unsafe_allow_html=True)
                                elif cv_std < 0.1:
                                    st.markdown('<div class="info-box">⚠️ Moderate variance in CV scores</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="error-box">❌ High variance in CV scores may indicate overfitting</div>', unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.markdown(f'<div class="error-box">❌ Error in cross-validation: {e}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                elif analysis_type == "Feature Importance":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Feature Importance Analysis")
                    
                    if ('Linear' in selected_model or 'Ridge' in selected_model or 'Lasso' in selected_model) and 'coefficients' in st.session_state.results[selected_model]:
                        try:
                            coefficients = st.session_state.results[selected_model]['coefficients']
                            feature_importance = pd.DataFrame({
                                'Feature': st.session_state.feature_names,
                                'Coefficient': coefficients,
                                'Abs_Coefficient': np.abs(coefficients)
                            }).sort_values('Abs_Coefficient', ascending=False)
                            
                            # Enhanced feature importance table
                            st.dataframe(feature_importance, use_container_width=True)
                            
                            # Enhanced feature importance plot
                            fig = px.bar(
                                feature_importance,
                                x='Abs_Coefficient',
                                y='Feature',
                                orientation='h',
                                title='Feature Importance (Absolute Coefficients)',
                                template="plotly_white",
                                color='Coefficient',
                                color_continuous_scale='RdBu'
                            )
                            fig.update_layout(
                                height=max(300, len(st.session_state.feature_names) * 40),
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                            
                            # Feature importance insights
                            most_important = feature_importance.iloc[0]
                            least_important = feature_importance.iloc[-1]
                            
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown(f"**🎯 Key Insights:**")
                            st.markdown(f"• Most important feature: **{most_important['Feature']}** (coef: {most_important['Coefficient']:.4f})")
                            st.markdown(f"• Least important feature: **{least_important['Feature']}** (coef: {least_important['Coefficient']:.4f})")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error in feature importance analysis: {e}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">Feature importance analysis is only available for linear, ridge, and lasso regression models.</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("🤖 **No models available for analysis!**")
                st.markdown("Train some models using the sidebar controls to perform advanced analysis here.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📁 Export & Download Center</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
                
                # Enhanced export options
                col1, col2 = st.columns([2, 1])
                with col1:
                    export_option = st.selectbox(
                        "Select export option:",
                        ["Generate Analysis Report", "Export Trained Model", "Download Dataset", "Export Visualizations"],
                        help="Choose what you want to export or download"
                    )
                with col2:
                    export_format = st.selectbox(
                        "Format:",
                        ["TXT", "JSON", "PDF"] if "Report" in export_option else ["PKL", "JSON"],
                        help="Select the export format"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if export_option == "Generate Analysis Report":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Comprehensive Analysis Report")
                    
                    # Report customization options
                    with st.expander("📝 Report Customization Options", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            include_data_summary = st.checkbox("Include data summary", value=True)
                            include_model_comparison = st.checkbox("Include model comparison", value=True)
                        with col2:
                            include_best_model = st.checkbox("Include best model details", value=True)
                            include_recommendations = st.checkbox("Include recommendations", value=True)
                        with col3:
                            include_technical_details = st.checkbox("Include technical details", value=False)
                            include_plots_description = st.checkbox("Include plots description", value=True)
                    
                    if st.button("📄 Generate Comprehensive Report", type="primary", use_container_width=True):
                        with st.spinner("📝 Generating detailed analysis report..."):
                            # Create comprehensive report
                            report = f"""
# REGRESSION ANALYSIS COMPREHENSIVE REPORT
{"=" * 80}
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
{"=" * 50}
This report presents a comprehensive analysis of regression models trained on the provided dataset.
The analysis includes {len(st.session_state.results)} different regression models with performance comparison and insights.

"""
                            
                            if include_data_summary:
                                report += f"""## DATASET INFORMATION
{"=" * 50}
• Dataset shape: {st.session_state.data.shape[0]} samples × {st.session_state.data.shape[1]} features
• Target variable: {st.session_state.target_name}
• Feature variables: {', '.join(st.session_state.feature_names)}
• Target statistics:
  - Mean: {st.session_state.y.mean():.4f}
  - Standard deviation: {st.session_state.y.std():.4f}
  - Range: {st.session_state.y.min():.4f} to {st.session_state.y.max():.4f}

### STATISTICAL SUMMARY
{st.session_state.data.describe().to_string()}

"""
                            
                            if include_model_comparison:
                                report += f"""## MODEL PERFORMANCE COMPARISON
{"=" * 50}
"""
                                for model_name, results in st.session_state.results.items():
                                    overfitting = abs(results['train_r2'] - results['test_r2'])
                                    report += f"""
### {model_name}
• Training R² Score: {results['train_r2']:.4f}
• Test R² Score: {results['test_r2']:.4f}
• Training MSE: {results['train_mse']:.4f}
• Test MSE: {results['test_mse']:.4f}
• Training RMSE: {results['train_rmse']:.4f}
• Test RMSE: {results['test_rmse']:.4f}
• Training MAE: {results['train_mae']:.4f}
• Test MAE: {results['test_mae']:.4f}
• Overfitting Indicator: {overfitting:.4f} {'(Good)' if overfitting < 0.05 else '(Moderate)' if overfitting < 0.1 else '(High)'}
"""
                            
                            if include_best_model:
                                best_model_name = max(st.session_state.results.items(), key=lambda x: x[1]['test_r2'])[0]
                                best_results = st.session_state.results[best_model_name]
                                
                                report += f"""
## BEST PERFORMING MODEL: {best_model_name}
{"=" * 50}
The {best_model_name} achieved the highest test R² score of {best_results['test_r2']:.4f}.

### Performance Metrics:
• Test R² Score: {best_results['test_r2']:.4f} (explains {best_results['test_r2']*100:.1f}% of variance)
• Test RMSE: {best_results['test_rmse']:.4f}
• Test MAE: {best_results['test_mae']:.4f}
• Generalization Gap: {abs(best_results['train_r2'] - best_results['test_r2']):.4f}

"""
                            
                            if include_recommendations:
                                report += f"""## RECOMMENDATIONS
{"=" * 50}
Based on the analysis results:

### Model Performance:
"""
                                best_r2 = max(result['test_r2'] for result in st.session_state.results.values())
                                if best_r2 > 0.8:
                                    report += "• Excellent model performance (R² > 0.8) - Models are highly predictive\n"
                                elif best_r2 > 0.6:
                                    report += "• Good model performance (R² > 0.6) - Models show strong predictive power\n"
                                elif best_r2 > 0.4:
                                    report += "• Moderate model performance (R² > 0.4) - Consider feature engineering or more complex models\n"
                                else:
                                    report += "• Low model performance (R² < 0.4) - Consider different features or non-linear approaches\n"
                                
                                # Check for overfitting
                                overfitting_models = [name for name, results in st.session_state.results.items() 
                                                    if abs(results['train_r2'] - results['test_r2']) > 0.1]
                                if overfitting_models:
                                    report += f"• Potential overfitting detected in: {', '.join(overfitting_models)}\n"
                                    report += "• Consider using regularization or reducing model complexity\n"
                                
                                report += "\n### Next Steps:\n"
                                report += "• Collect more data if possible to improve model generalization\n"
                                report += "• Try ensemble methods for better performance\n"
                                report += "• Consider feature selection/engineering for improved results\n"
                                report += "• Validate models on completely independent test sets\n"
                            
                            if include_technical_details:
                                report += f"""
## TECHNICAL DETAILS
{"=" * 50}
### Model Training Configuration:
• Train-test split ratio: 80%-20% (default)
• Cross-validation: K-fold with k=5 (if performed)
• Evaluation metrics: R², MSE, RMSE, MAE
• Feature scaling: Applied where necessary

### Data Quality Assessment:
• Missing values: {st.session_state.data.isnull().sum().sum()} total
• Data types: {dict(st.session_state.data.dtypes)}

"""
                            
                            report += f"""
## CONCLUSION
{"=" * 50}
The regression analysis successfully evaluated {len(st.session_state.results)} different models.
The best performing model ({max(st.session_state.results.items(), key=lambda x: x[1]['test_r2'])[0]}) 
achieved a test R² score of {max(result['test_r2'] for result in st.session_state.results.values()):.4f}.

For production deployment, consider the best performing model with appropriate monitoring and retraining schedules.

---
Report generated by Interactive Regression Simulator v1.0.0
"""
                            
                            # Display report in expandable text area
                            with st.expander("📋 Generated Report Preview", expanded=True):
                                st.text_area("Report Content:", report, height=400)
                            
                            # Enhanced download button
                            st.download_button(
                                label="📄 Download Complete Report",
                                data=report,
                                file_name=f"regression_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                type="primary"
                            )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                elif export_option == "Export Trained Model":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 🤖 Export Trained Model")
                    
                    selected_model = st.selectbox(
                        "Select model to export:",
                        list(st.session_state.results.keys()),
                        key="export_model",
                        help="Choose which trained model to export for deployment"
                    )
                    
                    # Model export options
                    with st.expander("⚙️ Export Options", expanded=True):
                        include_preprocessing = st.checkbox("Include preprocessing steps", value=True)
                        include_metadata = st.checkbox("Include model metadata", value=True)
                        include_validation_data = st.checkbox("Include validation results", value=True)
                    
                    if st.button("📦 Export Model Package", type="primary", use_container_width=True):
                        with st.spinner("📦 Preparing model export..."):
                            try:
                                model_data = {
                                    'model_name': selected_model,
                                    'model': st.session_state.models[selected_model],
                                    'feature_names': st.session_state.feature_names,
                                    'target_name': st.session_state.target_name,
                                    'model_results': st.session_state.results[selected_model],
                                    'export_timestamp': pd.Timestamp.now().isoformat(),
                                    'export_version': '1.0.0'
                                }
                                
                                if include_metadata:
                                    model_data['metadata'] = {
                                        'dataset_shape': st.session_state.data.shape,
                                        'feature_count': len(st.session_state.feature_names),
                                        'target_statistics': {
                                            'mean': float(st.session_state.y.mean()),
                                            'std': float(st.session_state.y.std()),
                                            'min': float(st.session_state.y.min()),
                                            'max': float(st.session_state.y.max())
                                        }
                                    }
                                
                                if include_validation_data:
                                    model_data['validation'] = {
                                        'train_r2': st.session_state.results[selected_model]['train_r2'],
                                        'test_r2': st.session_state.results[selected_model]['test_r2'],
                                        'test_rmse': st.session_state.results[selected_model]['test_rmse']
                                    }
                                
                                # Serialize model
                                buffer = io.BytesIO()
                                pickle.dump(model_data, buffer)
                                
                                st.markdown('<div class="success-box">✅ Model package prepared successfully!</div>', unsafe_allow_html=True)
                                
                                st.download_button(
                                    label="📦 Download Model Package",
                                    data=buffer.getvalue(),
                                    file_name=f"{selected_model.replace(' ', '_')}_model_package_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                    mime="application/octet-stream",
                                    use_container_width=True,
                                    type="primary"
                                )
                                
                            except Exception as e:
                                st.markdown(f'<div class="error-box">❌ Error preparing model export: {e}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                elif export_option == "Download Dataset":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Download Dataset")
                    
                    # Dataset export options
                    with st.expander("📋 Export Configuration", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            include_predictions = st.checkbox("Include model predictions", value=False)
                            include_residuals = st.checkbox("Include residuals", value=False)
                        with col2:
                            selected_model_for_data = st.selectbox(
                                "Model for predictions/residuals:",
                                list(st.session_state.results.keys()),
                                help="Choose model for adding predictions/residuals to dataset"
                            ) if (include_predictions or include_residuals) else None
                    
                    if st.button("📊 Prepare Dataset Download", type="primary", use_container_width=True):
                        with st.spinner("📊 Preparing dataset..."):
                            try:
                                export_data = st.session_state.data.copy()
                                
                                if include_predictions and selected_model_for_data:
                                    # Add predictions
                                    model = st.session_state.models[selected_model_for_data]
                                    if 'Polynomial' in selected_model_for_data:
                                        poly = model['poly']
                                        linear = model['linear']
                                        X_poly = poly.transform(st.session_state.X)
                                        predictions = linear.predict(X_poly)
                                    else:
                                        predictions = model.predict(st.session_state.X)
                                    
                                    export_data[f'{selected_model_for_data}_Predictions'] = predictions
                                
                                if include_residuals and selected_model_for_data:
                                    # Add residuals
                                    if f'{selected_model_for_data}_Predictions' in export_data.columns:
                                        residuals = export_data[st.session_state.target_name] - export_data[f'{selected_model_for_data}_Predictions']
                                    else:
                                        # Calculate predictions if not already added
                                        model = st.session_state.models[selected_model_for_data]
                                        if 'Polynomial' in selected_model_for_data:
                                            poly = model['poly']
                                            linear = model['linear']
                                            X_poly = poly.transform(st.session_state.X)
                                            predictions = linear.predict(X_poly)
                                        else:
                                            predictions = model.predict(st.session_state.X)
                                        residuals = st.session_state.y - predictions
                                    
                                    export_data[f'{selected_model_for_data}_Residuals'] = residuals
                                
                                csv = export_data.to_csv(index=False)
                                
                                st.markdown('<div class="success-box">✅ Dataset prepared for download!</div>', unsafe_allow_html=True)
                                st.markdown(f"**Dataset shape:** {export_data.shape[0]} rows × {export_data.shape[1]} columns")
                                
                                st.download_button(
                                    label="📊 Download Enhanced Dataset (CSV)",
                                    data=csv,
                                    file_name=f"regression_dataset_enhanced_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    type="primary"
                                )
                                
                            except Exception as e:
                                st.markdown(f'<div class="error-box">❌ Error preparing dataset: {e}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                elif export_option == "Export Visualizations":
                    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
                    st.markdown("#### 📈 Export Visualizations")
                    
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Note:** To export visualizations:")
                    st.markdown("1. Navigate to the visualization you want to export")
                    st.markdown("2. Use the camera icon 📷 in the top-right corner of each plot")
                    st.markdown("3. Choose your preferred format (PNG, SVG, PDF)")
                    st.markdown("4. Plots are saved with high resolution for publication quality")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Quick access to visualization tabs
                    st.markdown("**Quick Navigation:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📊 Data Visualizations", use_container_width=True):
                            st.info("Navigate to 'Data Explorer' tab for data visualizations")
                    with col2:
                        if st.button("🎯 Model Predictions", use_container_width=True):
                            st.info("Navigate to 'Model Predictions' tab for prediction plots")
                    with col3:
                        if st.button("🔍 Residual Plots", use_container_width=True):
                            st.info("Navigate to 'Residual Analysis' tab for diagnostic plots")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("🤖 **No models available for export!**")
                st.markdown("Train some models using the sidebar controls to access export options.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Enhanced welcome message when no data is loaded
        st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to the Interactive Regression Simulator!</h2>
            <p style="font-size: 1.2rem; color: #666; margin: 1.5rem 0;">
                🚀 A comprehensive tool for regression analysis, model comparison, and visualization
            </p>
            <p style="font-size: 1rem; color: #888; margin-bottom: 2rem;">
                Get started by loading data from the sidebar. You can use sample datasets, 
                upload your own CSV file, or enter data manually.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Data Analysis")
            st.markdown("• Multiple data sources")
            st.markdown("• Statistical summaries")
            st.markdown("• Interactive visualizations")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🤖 ML Models")
            st.markdown("• Linear Regression")
            st.markdown("• Polynomial Features")
            st.markdown("• Ridge & Lasso")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Visualizations")
            st.markdown("• Model predictions")
            st.markdown("• Residual analysis")
            st.markdown("• Feature importance")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 📁 Export Tools")
            st.markdown("• Comprehensive reports")
            st.markdown("• Model packages")
            st.markdown("• Enhanced datasets")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced footer with deployment and version info
    st.markdown('<div class="footer-section">', unsafe_allow_html=True)
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.markdown("### 🚀 Deployment Ready")
        st.markdown("✅ Streamlit Cloud compatible")
        st.markdown("✅ Docker containerized")
        st.markdown("✅ Heroku deployable")
    with footer_col2:
        st.markdown("### 🔧 Technical Stack")
        st.markdown("• **Framework:** Streamlit")
        st.markdown("• **ML Library:** scikit-learn")
        st.markdown("• **Visualization:** Plotly")
        st.markdown("• **Data:** Pandas, NumPy")
    with footer_col3:
        st.markdown("### 📊 Features")
        st.markdown("• Real-time model training")
        st.markdown("• Interactive predictions")
        st.markdown("• Advanced analytics")
        st.markdown("• Export capabilities")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Version and performance info
    st.markdown('<div class="version-info">', unsafe_allow_html=True)
    st.markdown("**Interactive Regression Simulator v1.0.0** | 🎨 Enhanced UI | ⚡ Optimized for deployment")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.markdown("Please refresh the page or contact support.")
        
        # In development, show the full error
        if st.secrets.get("environment", "production") == "development":
            st.exception(e)
