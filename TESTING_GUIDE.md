# NextTrack API - Manual Testing Guide

This guide provides step-by-step instructions for manually testing the NextTrack API functionality.

## Prerequisites

Before running any tests, ensure you have:

1. **Python 3.10+ installed**
2. **Virtual environment activated**:
   ```bash
   cd "Final Version"
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **All dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Internet connection** (for MusicBrainz API)

## Quick Test (Automated)

For a comprehensive automated test, simply run:

```bash
python tests/test_suites.py
```

This will run all tests automatically and provide a detailed report.

## Manual Testing Steps

If you prefer to run tests manually, follow these steps:

### Step 1: Test Data Generation

```bash
# Test generating real music data from MusicBrainz
python scripts/generate.py --tracks 20 --output test_data --models-dir test_data/models
```

**Expected outcome**:

- Creates `test_data/` directory
- Downloads 20 real tracks from MusicBrainz
- Generates user sessions
- Trains LightFM model
- Creates content-based features

**Verify**: Check that these files exist:

- `test_data/production_tracks.json`
- `test_data/production_track_metadata.json`
- `test_data/production_sessions.json`
- `test_data/models/lightfm_model.pkl`
- `test_data/models/cb_features.pkl`

### Step 2: Test API Startup

```bash
# Test API can start and load models
python -c "
import os
import sys
sys.path.append('.')

# Configure for testing
os.environ['USE_ANN_INDEX'] = 'false'
os.environ['USE_CONTENT_BASED'] = 'true'
os.environ['DATA_DIR'] = 'test_data'

from api.main import app
from fastapi.testclient import TestClient

print('🚀 Testing API startup...')
with TestClient(app) as client:
    response = client.get('/health')
    print(f'Health check: {response.status_code}')
    if response.status_code == 200:
        health = response.json()
        print(f'API Status: {health.get(\"status\")}')
        print('✅ API startup successful!')
    else:
        print('❌ API startup failed!')
"
```

**Expected outcome**: API starts successfully with "healthy" status

### Step 3: Test Recommendations

```bash
# Test recommendation functionality
python -c "
import os
import sys
import json
sys.path.append('.')

os.environ['USE_ANN_INDEX'] = 'false'
os.environ['USE_CONTENT_BASED'] = 'true'
os.environ['DATA_DIR'] = 'test_data'

from api.main import app
from fastapi.testclient import TestClient

print('🎵 Testing recommendations...')
with TestClient(app) as client:
    # Load test tracks
    with open('test_data/production_tracks.json', 'r') as f:
        tracks = json.load(f)

    track_ids = [track['track_id'] for track in tracks[:2]]
    print(f'Using tracks: {[track[\"title\"] for track in tracks[:2]]}')

    # Request recommendations
    response = client.post('/recommendations', json={
        'track_sequence': track_ids,
        'top_k': 5,
        'fusion_method': 'weighted_sum'
    })

    if response.status_code == 200:
        data = response.json()
        print(f'✅ Got {len(data[\"recommendations\"])} recommendations')
        print(f'Fusion method: {data[\"metadata\"][\"fusion_method\"]}')
        print(f'CF weight: {data[\"metadata\"][\"cf_weight\"]}')
        print(f'CB weight: {data[\"metadata\"][\"cb_weight\"]}')
    else:
        print(f'❌ Recommendations failed: {response.status_code}')
        print(f'Error: {response.text}')
"
```

**Expected outcome**: Returns 5 track recommendations with metadata about the fusion method

### Step 4: Test Track Listing

```bash
# Test track listing and pagination
python -c "
import os
import sys
sys.path.append('.')

os.environ['DATA_DIR'] = 'test_data'

from api.main import app
from fastapi.testclient import TestClient

print('📋 Testing track listing...')
with TestClient(app) as client:
    response = client.get('/tracks?page=1&page_size=5')

    if response.status_code == 200:
        data = response.json()
        tracks = data.get('tracks', [])
        print(f'✅ Retrieved {len(tracks)} tracks')
        if tracks:
            print(f'Sample track: \"{tracks[0][\"title\"]}\" by {tracks[0][\"artist\"]}')
        print(f'Total tracks: {data.get(\"total\", 0)}')
        print(f'Current page: {data.get(\"page\", 0)}')
    else:
        print(f'❌ Track listing failed: {response.status_code}')
"
```

**Expected outcome**: Returns paginated list of tracks with metadata

### Step 5: Test API Documentation

```bash
# Test Swagger UI is accessible
python -c "
import os
import sys
sys.path.append('.')

os.environ['DATA_DIR'] = 'test_data'

from api.main import app
from fastapi.testclient import TestClient

print('📚 Testing API documentation...')
with TestClient(app) as client:
    # Test Swagger UI
    response = client.get('/docs')
    print(f'Swagger UI: {\"✅ Available\" if response.status_code == 200 else \"❌ Error\"}')

    # Test OpenAPI spec
    response = client.get('/openapi.json')
    if response.status_code == 200:
        spec = response.json()
        endpoints = len(spec.get('paths', {}))
        print(f'✅ OpenAPI spec available with {endpoints} endpoints')
    else:
        print('❌ OpenAPI spec not available')
"
```

**Expected outcome**: Swagger UI accessible at `/docs` with interactive API documentation

### Step 6: Run Unit Tests

```bash
# Run the complete test suite
pytest tests/test_api.py -v
```

**Expected outcome**: All tests pass (typically 13/13 tests)

### Step 7: Clean Up

```bash
# Remove test data
rm -rf test_data/
```

## Performance Verification

To verify the system is working optimally:

1. **Check startup time**: API should start in < 5 seconds
2. **Check response time**: Recommendations should return in < 100ms
3. **Check memory usage**: Should use < 200MB for small datasets
4. **Check real data**: Verify tracks have real artist names and titles (not "Track 001")

## Troubleshooting

### Common Issues and Solutions

**Issue**: `ModuleNotFoundError: No module named 'training'`

```bash
# Solution: Add current directory to Python path or run from project root
cd "Final Version"
python manual_test_script.py
```

**Issue**: `MusicBrainz API connection failed`

```bash
# Solution: Check internet connection and try again
# MusicBrainz has rate limiting, wait a moment between requests
```

**Issue**: `LightFM import error`

```bash
# Solution: Install LightFM
pip install lightfm
```

**Issue**: `Redis connection error`

```bash
# Solution: This is normal if Redis is not installed
# The system works without Redis (caching disabled)
```

**Issue**: `FAISS import error`

```bash
# Solution: Install FAISS or disable ANN indexing
pip install faiss-cpu
# OR
export USE_ANN_INDEX=false
```

## Success Criteria

The NextTrack API is working correctly if:

1. ✅ Data generation fetches real tracks from MusicBrainz
2. ✅ API starts without errors and reports "healthy" status
3. ✅ Recommendations return relevant track IDs
4. ✅ Track listing shows real track titles and artists
5. ✅ Swagger UI documentation is accessible
6. ✅ All unit tests pass
7. ✅ System handles errors gracefully

## Next Steps

Once all tests pass:

1. **Start the development server**:

   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the interactive docs**: http://localhost:8000/docs

3. **Try live recommendations** with real MusicBrainz track IDs

4. **Deploy to production** using Docker if needed
