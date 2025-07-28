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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 12px;
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
    
    # Header with deployment info
    st.markdown('<h1 class="main-header">🧠 Interactive Regression Simulator</h1>', unsafe_allow_html=True)
    st.markdown("**A comprehensive tool for regression analysis, model comparison, and visualization**")
    
    # Add deployment status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("*Deploy-ready Streamlit application*")
    with col2:
        st.markdown(f"**v{st.session_state.get('app_version', '1.0.0')}**")
    with col3:
        if st.button("🔄 Reset App"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🎛️ Control Panel</h2>', unsafe_allow_html=True)
        
        # Data Section
        st.markdown("### 📊 Data Management")
        
        # Data loading options
        data_option = st.selectbox(
            "Choose data source:",
            ["Select...", "Sample Datasets", "Upload CSV", "Manual Entry"]
        )
        
        if data_option == "Sample Datasets":
            dataset_type = st.selectbox(
                "Select sample dataset:",
                ["Housing Data", "Simple Linear", "Nonlinear Data", "Boston Housing"]
            )
            
            # Add sample size selector for performance
            n_samples = st.slider("Number of samples:", 50, 1000, 200, 50)
            
            if st.button("Load Sample Data", type="primary"):
                with st.spinner("Loading data..."):
                    success = app.load_sample_data(dataset_type)
                    if success:
                        st.success(f"✅ {dataset_type} loaded successfully!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
        
        elif data_option == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Add file size check for deployment
                    file_size = uploaded_file.size / (1024 * 1024)  # MB
                    if file_size > 10:  # 10MB limit
                        st.error("File size too large! Please upload a file smaller than 10MB.")
                    else:
                        data = pd.read_csv(uploaded_file)
                        st.write("Preview:")
                        st.dataframe(data.head())
                        
                        # Column selection
                        columns = data.columns.tolist()
                        target_col = st.selectbox("Select target variable:", columns)
                        feature_cols = st.multiselect("Select feature variables:", 
                                                    [col for col in columns if col != target_col])
                        
                        if st.button("Load CSV Data", type="primary") and feature_cols:
                            # Validate data
                            if len(feature_cols) == 0:
                                st.error("Please select at least one feature!")
                            elif data[feature_cols + [target_col]].isnull().sum().sum() > 0:
                                st.warning("Data contains missing values. They will be dropped.")
                                data = data[feature_cols + [target_col]].dropna()
                            
                            st.session_state.data = data
                            st.session_state.feature_names = feature_cols
                            st.session_state.target_name = target_col
                            st.session_state.X = data[feature_cols].values
                            st.session_state.y = data[target_col].values
                            
                            # Clear previous models
                            st.session_state.models = {}
                            st.session_state.results = {}
                            
                            st.success("✅ CSV data loaded successfully!")
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        # Model Training Section
        if st.session_state.data is not None:
            st.markdown("### 🤖 Model Training")
            
            model_type = st.selectbox(
                "Select model type:",
                ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression"]
            )
            
            # Advanced options expander
            with st.expander("⚙️ Advanced Options"):
                test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
                random_state = st.number_input("Random state:", 0, 1000, 42)
            
            # Model-specific parameters
            model_params = {'test_size': test_size, 'random_state': random_state}
            
            if model_type == "Polynomial Regression":
                degree = st.slider("Polynomial degree:", 2, 10, 2)
                model_params['degree'] = degree
                
                # Warning for high degrees
                if degree > 5:
                    st.warning("⚠️ High polynomial degrees may cause overfitting!")
                    
            elif model_type in ["Ridge Regression", "Lasso Regression"]:
                alpha = st.slider("Regularization strength (α):", 0.001, 10.0, 1.0, step=0.001, format="%.3f")
                model_params['alpha'] = alpha
                
                if alpha < 0.01:
                    st.info("💡 Very low α values may lead to overfitting")
                elif alpha > 5:
                    st.info("💡 High α values may lead to underfitting")
            
            if st.button("Train Model", type="primary"):
                model_key = model_type.split()[0]  # Linear, Polynomial, Ridge, Lasso
                
                with st.spinner(f"Training {model_type}..."):
                    model_name, results = app.train_model(model_key, **model_params)
                    
                    if model_name and results:
                        st.success(f"✅ {model_name} trained successfully!")
                        
                        # Show quick metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Test R²", f"{results['test_r2']:.4f}")
                        with col2:
                            st.metric("Test RMSE", f"{results['test_rmse']:.2f}")
                        
                        time.sleep(1)
                        st.rerun()
            
            # Quick model comparison
            if len(st.session_state.results) > 0:
                st.markdown("### 📊 Quick Comparison")
                
                best_model = max(st.session_state.results.items(), key=lambda x: x[1]['test_r2'])
                st.success(f"🏆 Best: {best_model[0]} (R² = {best_model[1]['test_r2']:.4f})")
                
                st.write(f"**Models trained:** {len(st.session_state.results)}")
        
        # Footer with deployment info
        st.markdown("---")
        st.markdown("### 🚀 Deployment Ready")
        st.markdown("This app is optimized for:")
        st.markdown("- Streamlit Cloud")
        st.markdown("- Heroku")
        st.markdown("- Docker")
        st.markdown("- AWS/GCP")
    
    # Main content area
    if st.session_state.data is not None:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🎯 Model Results", 
            "📈 Visualizations", 
            "🔍 Analysis", 
            "📋 Export"
        ])
        
        with tab1:
            st.markdown('<h3 class="sub-header">Dataset Information</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", st.session_state.data.shape[0])
            with col2:
                st.metric("Features", len(st.session_state.feature_names))
            with col3:
                st.metric("Target", st.session_state.target_name)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head(10))
            
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(st.session_state.data.describe())
            
            # Data visualization
            st.subheader("Data Visualization")
            fig = app.create_data_visualization()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown('<h3 class="sub-header">Model Performance</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                # Model comparison table
                comparison_data = []
                for model_name, results in st.session_state.results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Train R²': f"{results['train_r2']:.4f}",
                        'Test R²': f"{results['test_r2']:.4f}",
                        'Train MSE': f"{results['train_mse']:.4f}",
                        'Test MSE': f"{results['test_mse']:.4f}"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                # Model comparison chart
                st.subheader("Model Comparison Chart")
                fig = app.create_model_comparison_plot()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No models trained yet. Please train some models first.")
        
        with tab3:
            st.markdown('<h3 class="sub-header">Model Visualizations</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                selected_model = st.selectbox(
                    "Select model to visualize:",
                    list(st.session_state.results.keys())
                )
                
                if selected_model:
                    # Model predictions
                    st.subheader("Model Predictions")
                    fig = app.create_model_predictions_plot(selected_model)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Interactive prediction
                    st.subheader("Interactive Prediction")
                    st.write("Enter feature values to get a prediction:")
                    
                    prediction_values = []
                    cols = st.columns(len(st.session_state.feature_names))
                    
                    for i, feature in enumerate(st.session_state.feature_names):
                        with cols[i]:
                            min_val = float(st.session_state.data[feature].min())
                            max_val = float(st.session_state.data[feature].max())
                            default_val = float(st.session_state.data[feature].mean())
                            
                            value = st.number_input(
                                f"{feature}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"pred_{feature}"
                            )
                            prediction_values.append(value)
                    
                    if st.button("Make Prediction"):
                        features = np.array(prediction_values).reshape(1, -1)
                        
                        if 'Polynomial' in selected_model:
                            poly = st.session_state.models[selected_model]['poly']
                            linear = st.session_state.models[selected_model]['linear']
                            features_poly = poly.transform(features)
                            prediction = linear.predict(features_poly)[0]
                        else:
                            model = st.session_state.models[selected_model]
                            prediction = model.predict(features)[0]
                        
                        st.success(f"🎯 Predicted {st.session_state.target_name}: **{prediction:.2f}**")
            else:
                st.info("No models available for visualization. Please train some models first.")
        
        with tab4:
            st.markdown('<h3 class="sub-header">Advanced Analysis</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                analysis_type = st.selectbox(
                    "Select analysis type:",
                    ["Residual Analysis", "Cross-Validation", "Feature Importance"]
                )
                
                selected_model = st.selectbox(
                    "Select model for analysis:",
                    list(st.session_state.results.keys()),
                    key="analysis_model"
                )
                
                if analysis_type == "Residual Analysis":
                    st.subheader("Residual Analysis")
                    fig = app.create_residual_analysis(selected_model)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Residual statistics
                    results = st.session_state.results[selected_model]
                    train_residuals = results['y_train'] - results['y_train_pred']
                    test_residuals = results['y_test'] - results['y_test_pred']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train Residual Mean", f"{np.mean(train_residuals):.4f}")
                        st.metric("Train Residual Std", f"{np.std(train_residuals):.4f}")
                    with col2:
                        st.metric("Test Residual Mean", f"{np.mean(test_residuals):.4f}")
                        st.metric("Test Residual Std", f"{np.std(test_residuals):.4f}")
                
                elif analysis_type == "Cross-Validation":
                    st.subheader("Cross-Validation Analysis")
                    
                    n_folds = st.slider("Number of folds:", 3, 10, 5)
                    
                    if st.button("Run Cross-Validation"):
                        with st.spinner("Running cross-validation..."):
                            if 'Linear' in selected_model:
                                model = LinearRegression()
                                cv_scores = cross_val_score(model, st.session_state.X, st.session_state.y, cv=n_folds, scoring='r2')
                            elif 'Polynomial' in selected_model:
                                degree = st.session_state.results[selected_model]['degree']
                                poly = PolynomialFeatures(degree=degree)
                                X_poly = poly.fit_transform(st.session_state.X)
                                model = LinearRegression()
                                cv_scores = cross_val_score(model, X_poly, st.session_state.y, cv=n_folds, scoring='r2')
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean CV Score", f"{np.mean(cv_scores):.4f}")
                            with col2:
                                st.metric("Std CV Score", f"{np.std(cv_scores):.4f}")
                            with col3:
                                st.metric("Min CV Score", f"{np.min(cv_scores):.4f}")
                            
                            # CV scores plot
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                                y=cv_scores,
                                marker_color='lightblue'
                            ))
                            fig.update_layout(
                                title=f"{n_folds}-Fold Cross-Validation Scores",
                                xaxis_title="Fold",
                                yaxis_title="R² Score",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                elif analysis_type == "Feature Importance":
                    if 'Linear' in selected_model and 'coefficients' in st.session_state.results[selected_model]:
                        st.subheader("Feature Importance")
                        
                        coefficients = st.session_state.results[selected_model]['coefficients']
                        feature_importance = pd.DataFrame({
                            'Feature': st.session_state.feature_names,
                            'Coefficient': coefficients,
                            'Abs_Coefficient': np.abs(coefficients)
                        }).sort_values('Abs_Coefficient', ascending=False)
                        
                        st.dataframe(feature_importance)
                        
                        # Feature importance plot
                        fig = px.bar(
                            feature_importance,
                            x='Feature',
                            y='Abs_Coefficient',
                            title='Feature Importance (Absolute Coefficients)',
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Feature importance analysis is only available for linear models.")
            else:
                st.info("No models available for analysis. Please train some models first.")
        
        with tab5:
            st.markdown('<h3 class="sub-header">Export Options</h3>', unsafe_allow_html=True)
            
            if st.session_state.results:
                export_option = st.selectbox(
                    "Select export option:",
                    ["Generate Report", "Export Model", "Download Data"]
                )
                
                if export_option == "Generate Report":
                    st.subheader("Generate Analysis Report")
                    
                    if st.button("Generate Report"):
                        # Create comprehensive report
                        report = f"""
# REGRESSION ANALYSIS REPORT
{"=" * 80}

## Dataset Information
- Number of samples: {st.session_state.data.shape[0]}
- Features: {', '.join(st.session_state.feature_names)}
- Target variable: {st.session_state.target_name}

## Statistical Summary
{st.session_state.data.describe().to_string()}

## Model Performance Comparison
"""
                        for model_name, results in st.session_state.results.items():
                            report += f"""
### {model_name}
- Training R²: {results['train_r2']:.4f}
- Test R²: {results['test_r2']:.4f}
- Training MSE: {results['train_mse']:.4f}
- Test MSE: {results['test_mse']:.4f}
"""
                        
                        st.text_area("Generated Report:", report, height=400)
                        
                        # Download button
                        st.download_button(
                            label="📄 Download Report",
                            data=report,
                            file_name="regression_analysis_report.txt",
                            mime="text/plain"
                        )
                
                elif export_option == "Export Model":
                    st.subheader("Export Trained Model")
                    
                    selected_model = st.selectbox(
                        "Select model to export:",
                        list(st.session_state.results.keys()),
                        key="export_model"
                    )
                    
                    if st.button("Export Model"):
                        model_data = {
                            'name': selected_model,
                            'model': st.session_state.models[selected_model],
                            'feature_names': st.session_state.feature_names,
                            'target_name': st.session_state.target_name,
                            'results': st.session_state.results[selected_model]
                        }
                        
                        # Serialize model
                        buffer = io.BytesIO()
                        pickle.dump(model_data, buffer)
                        
                        st.download_button(
                            label="📦 Download Model",
                            data=buffer.getvalue(),
                            file_name=f"{selected_model.replace(' ', '_')}_model.pkl",
                            mime="application/octet-stream"
                        )
                
                elif export_option == "Download Data":
                    st.subheader("Download Dataset")
                    
                    csv = st.session_state.data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv,
                        file_name="regression_data.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No models available for export. Please train some models first.")
    
    else:
        # Welcome message when no data is loaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to the Interactive Regression Simulator!</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Get started by loading some data from the sidebar. 
                You can use sample datasets, upload your own CSV file, or enter data manually.
            </p>
            <p style="font-size: 1rem; color: #888;">
                Once you load data, you'll be able to train multiple regression models, 
                visualize results, and perform advanced analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add footer with deployment and version info
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.markdown("**🚀 Deployment Ready**")
        st.markdown("Optimized for cloud platforms")
    with footer_col2:
        st.markdown("**📊 Performance**")
        if st.session_state.data is not None:
            st.markdown(f"Dataset: {st.session_state.data.shape[0]} samples")
            st.markdown(f"Models: {len(st.session_state.results)} trained")
        else:
            st.markdown("No data loaded")
    with footer_col3:
        st.markdown("**ℹ️ Info**")
        st.markdown(f"Version: {st.session_state.get('app_version', '1.0.0')}")
        st.markdown("[📖 Documentation](https://github.com/your-repo)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.markdown("Please refresh the page or contact support.")
        
        # In development, show the full error
        if st.secrets.get("environment", "production") == "development":
            st.exception(e)
