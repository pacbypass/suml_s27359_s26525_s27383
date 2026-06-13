#!/bin/bash
# Start the Flask web app for cat/dog breed classification

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in ./venv..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing/updating dependencies..."
pip install -r requirements.txt

# Run the app
echo "Starting Flask app..."
python main.py