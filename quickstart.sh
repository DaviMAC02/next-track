#!/bin/bash

# NextTrack API Quick Start Script
# This script helps you get the NextTrack API up and running quickly

set -e  # Exit on any error

echo "🎵 NextTrack API Quick Start 🎵"
echo "================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the NextTrack project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate sample data
generate_sample_data() {
    echo "🏗️  Generating sample data..."
    echo "   📝 Creating 100 sample tracks and user sessions..."
    echo "   🤖 Training recommendation models..."
    echo "   ⏱️  This may take 2-5 minutes..."
    echo ""
    
    # Show progress
    python3 scripts/generate.py --tracks 100 --output data --models-dir data/models &
    GENERATE_PID=$!
    
    # Simple progress indicator
    while kill -0 $GENERATE_PID 2>/dev/null; do
        printf "."
        sleep 2
    done
    
    # Wait for the process to complete and get exit status
    wait $GENERATE_PID
    return $?
}

# Check Python version
echo "📋 Checking prerequisites..."
if ! command_exists python3; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file. You can modify it if needed."
else
    echo "✅ .env file already exists"
fi

# Check if data files exist and offer to generate them
echo "📊 Checking data files..."

MISSING_DATA=false
REQUIRED_FILES=("data/lightfm_embeddings.json" "data/metadata_features.npy" "data/production_tracks.json" "data/production_sessions.json")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: $file not found"
        MISSING_DATA=true
    fi
done

if [ "$MISSING_DATA" = true ]; then
    echo ""
    echo "❓ Some required data files are missing."
    read -p "🏗️  Would you like to generate sample data now? This will take a few minutes. (Y/n): " generate_data
    
    if [[ ! $generate_data =~ ^[Nn]$ ]]; then
        echo ""
        
        # Create data directory if it doesn't exist
        mkdir -p data/models
        
        # Generate data with progress indication
        if generate_sample_data; then
            echo ""
            echo "✅ Sample data generated successfully!"
            echo "   📁 Generated files in data/ directory"
            echo "   🤖 Trained models in data/models/ directory"
            echo "   📊 Generated 100 tracks with user sessions and recommendations"
        else
            echo ""
            echo "❌ Failed to generate sample data"
            echo "   The API may not work properly without data files"
            echo "   You can try running manually: python3 scripts/generate.py --help"
            echo ""
            read -p "❓ Continue anyway? (y/N): " continue_anyway
            if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
                echo "Exiting. Run this script again when ready."
                exit 1
            fi
        fi
    else
        echo "⚠️  Skipping data generation."
        echo "   Note: The API may not work properly without required data files"
        echo "   You can generate data later by running: python3 scripts/generate.py"
        echo ""
        read -p "❓ Continue anyway? (y/N): " continue_anyway
        if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
            echo "Exiting. Run this script again when ready to generate data."
            exit 1
        fi
    fi
else
    echo "✅ All required data files found"
fi

# Check if we can import the main modules
echo "🧪 Testing imports..."
python3 -c "
try:
    from api.main import app
    from api.hybrid_recommender import HybridRecommender
    from config.settings import settings
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Offer to run tests
echo ""
read -p "🧪 Would you like to run the test suite? (y/N): " run_tests
if [[ $run_tests =~ ^[Yy]$ ]]; then
    echo "🧪 Running tests..."
    python3 -m pytest tests/ -v
    
    # Clean up test data after tests
    echo ""
    echo "🧹 Cleaning up test data..."
    
    # Remove any temporary test files but keep the main data directory
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache 2>/dev/null || true
    rm -f .coverage 2>/dev/null || true
    
    # Remove temporary test data directory created during tests but preserve data/
    if [ -d "test_data" ]; then
        rm -rf test_data
    fi
    
    echo "✅ Test cleanup complete - data folder preserved"
fi

# Offer to start the API
echo ""
read -p "🚀 Would you like to start the API now? (Y/n): " start_api
if [[ ! $start_api =~ ^[Nn]$ ]]; then
    echo ""
    echo "🚀 Starting NextTrack API..."
    echo "   API will be available at: http://localhost:8000"
    echo "   Interactive docs at: http://localhost:8000/docs"
    echo "   Health check at: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Start the API
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo ""
    echo "✅ Setup complete! To start the API manually, run:"
    echo "   source .venv/bin/activate"
    echo "   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
    echo "📚 Useful commands:"
    echo "   • Health check: curl http://localhost:8000/health"
    echo "   • Run tests: python3 -m pytest"
    echo "   • Generate sample data: python3 scripts/generate.py --tracks 100"
    echo "   • Run evaluation: python3 evaluate.py --sessions data/production_sessions.json"
    echo "   • View logs: tail -f logs/*.log"
fi
