# 🧠 Regression Simulator

A comprehensive Python-based regression analysis tool that provides an interactive environment for training, evaluating, and comparing multiple regression models with advanced visualization and reporting capabilities.

---

## 🚀 Features

### 📊 Data Management
- **Load Sample Datasets**: Built-in datasets (Boston Housing, California Housing)
- **CSV File Import**: Load custom datasets from CSV files
- **Manual Data Entry**: Enter data points manually for small datasets
- **Data Preprocessing**: Automatic feature scaling and train-test splitting

### 🤖 Model Training & Analysis
- **Multiple Regression Models**: Linear, Polynomial, Ridge, Lasso, and more
- **Cross-Validation Analysis**: K-fold cross-validation with customizable folds
- **Learning Curve Analysis**: Analyze model performance vs training size
- **Feature Importance Analysis**: Understand which features matter most
- **Residual Analysis**: Comprehensive residual plots and statistical tests

### 📈 Visualization
- **Interactive 2D Explorer**: Real-time model visualization and data exploration
- **Model Comparison Plots**: Side-by-side performance comparisons
- **Learning Curves**: Bias-variance analysis through learning curves
- **Residual Plots**: Scatter plots, Q-Q plots, and distribution analysis
- **Feature Importance Charts**: Visual representation of feature contributions

### 📝 Reporting & Export
- **Comprehensive Reports**: Automatically generated analysis reports with model formulas, performance metrics, and interpretations
- **Model Export**: Save trained models using pickle for future use
- **Performance Summaries**: Detailed statistical summaries and correlations
- **Export Formats**: Text reports and pickled model files

---

## 🛠 Requirements

### Dependencies
Install all required dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib colorama seaborn statsmodels
```

### Individual Package Descriptions
- **numpy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib**: Plotting and visualization
- **colorama**: Cross-platform colored terminal text
- **seaborn**: Statistical data visualization
- **statsmodels**: Statistical modeling and tests

### Python Version
- Python 3.7 or higher recommended

---

## 🎯 Getting Started

### 🌐 Web Application (Recommended)
**Interactive Web Interface with Modern UI:**

1. **Quick Start:**
   ```bash
   cd d:\VS_CODE\College_Assignment\ISL\visualizer
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **One-Click Run (Windows):**
   ```bash
   run_app.bat
   ```

3. **One-Click Run (Linux/Mac):**
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

4. **Access the app:** Open http://localhost:8501 in your browser

### 💻 Command Line Interface
**Traditional Terminal-Based Interface:**

1. **Create and run the CLI version:**
   ```python
   from visualizer import RegressionSimulator
   simulator = RegressionSimulator()
   simulator.main_menu()
   ```

2. **Or use the Jupyter Notebook:**
   ```bash
   jupyter notebook regression.ipynb
   ```

### 🚀 Online Deployment
Deploy your own instance online:
- **Streamlit Cloud:** Free hosting at share.streamlit.io
- **Heroku:** Cloud platform deployment
- **Docker:** Containerized deployment
- See `DEPLOYMENT.md` for detailed instructions

---

## 📋 Usage Guide

### 1. Loading Data
- **Option 1**: Use built-in datasets (recommended for beginners)
- **Option 2**: Load your own CSV file with headers
- **Option 3**: Manually enter small datasets

### 2. Training Models
- Select from various regression algorithms
- Configure model parameters (e.g., polynomial degree)
- Compare multiple models simultaneously

### 3. Analysis Options
- **Residual Analysis**: Check model assumptions
- **Cross-Validation**: Robust performance evaluation
- **Learning Curves**: Diagnose overfitting/underfitting
- **Feature Importance**: Understand predictor variables

### 4. Visualization
- Interactive 2D plots for model exploration
- Performance comparison charts
- Statistical diagnostic plots

### 5. Export Results
- Generate comprehensive analysis reports
- Export trained models for production use
- Save visualizations and summaries

---

## 📊 Sample Workflow

1. **Load Data** → Choose sample dataset or upload CSV
2. **Explore Data** → View statistics and correlations
3. **Train Models** → Compare Linear, Polynomial, Ridge regression
4. **Analyze Results** → Check residuals and cross-validation scores
5. **Generate Report** → Export comprehensive analysis
6. **Export Model** → Save best-performing model

---

## 🔧 Advanced Features

### Statistical Tests
- Durbin-Watson test for autocorrelation
- Normality tests for residuals
- Homoscedasticity analysis

### Model Diagnostics
- Q-Q plots for residual analysis
- Cook's distance for outlier detection
- Leverage and influence analysis

### Interactive Elements
- Real-time parameter adjustment
- Dynamic plot updates
- User-guided analysis workflow

---

## 📁 File Structure

```
visualizer/
├── visualizer.py          # Main application code
├── run_simulator.py       # Entry point script
├── ReadMe.txt            # This file
└── exports/              # Generated reports and models
    ├── reports/          # Analysis reports (.txt)
    └── models/           # Saved models (.pkl)
```

---

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Data Loading Issues**: Check CSV file format and headers
3. **Visualization Problems**: Update matplotlib backend if plots don't display
4. **Memory Issues**: Use smaller datasets or reduce cross-validation folds

### Getting Help
- Check error messages for specific guidance
- Ensure Python 3.7+ is being used
- Verify all dependencies are correctly installed
- Try with built-in sample data first

---

## 🎓 Educational Value

This tool is designed for:
- **Students**: Learning regression concepts and model comparison
- **Researchers**: Quick prototype and analysis of regression problems
- **Data Scientists**: Model validation and performance benchmarking
- **Educators**: Teaching machine learning and statistical modeling

---

## 📈 Future Enhancements

- Support for classification problems
- More advanced regression algorithms
- Enhanced visualization options
- Web-based interface
- Automatic hyperparameter tuning

---

*Built with Python for educational and research purposes. Perfect for understanding regression analysis and model comparison techniques.*