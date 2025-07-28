#!/bin/bash
echo "Starting Interactive Regression Simulator..."
echo ""
echo "Installing/Updating dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""
streamlit run app.py
