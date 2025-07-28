#!/bin/bash

# Deployment script for Regression Simulator
echo "🚀 Deploying Interactive Regression Simulator..."

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Install all dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    exit 1
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Copy secrets template if secrets.toml doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    if [ -f ".streamlit/secrets_template.toml" ]; then
        cp .streamlit/secrets_template.toml .streamlit/secrets.toml
        echo "⚠️  Please edit .streamlit/secrets.toml with your configuration"
    fi
fi

# Run the application
echo "🎯 Starting Regression Simulator..."
echo "📱 Open http://localhost:8501 in your browser"
echo "🛑 Press Ctrl+C to stop"

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
