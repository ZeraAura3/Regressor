@echo off
echo 🚀 Starting Interactive Regression Simulator...
echo.
echo 📦 Installing/Updating dependencies...
pip install -r requirements.txt
echo.
echo 🎯 Starting Streamlit app...
echo 📱 The app will open in your browser at http://localhost:8501
echo 🛑 Press Ctrl+C to stop the application
echo.

REM Create .streamlit directory if it doesn't exist
if not exist ".streamlit" mkdir .streamlit

REM Copy secrets template if secrets.toml doesn't exist
if not exist ".streamlit\secrets.toml" (
    if exist ".streamlit\secrets_template.toml" (
        copy ".streamlit\secrets_template.toml" ".streamlit\secrets.toml"
        echo ⚠️  Please edit .streamlit\secrets.toml with your configuration
    )
)

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
