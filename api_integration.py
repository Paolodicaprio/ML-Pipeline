#!/usr/bin/env python3
"""
Integration script to connect the FastAPI with the existing ML pipeline.
This script modifies the pipeline to automatically reload the API when v_best is updated.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_api_with_pipeline():
    """
    Integrate the API with the existing ML pipeline by modifying deployment scripts.
    """
    logger.info("Integrating API with ML pipeline...")
    
    # Check if deploy_model.py exists
    deploy_script = Path("src/deploy_model.py")
    if not deploy_script.exists():
        logger.error("deploy_model.py not found in src/ directory")
        return False
    
    # Read the existing deploy script
    with open(deploy_script, 'r') as f:
        deploy_content = f.read()
    
    # Add API notification to the deploy script
    api_integration_code = '''
# API Integration - Notify API of model update
def notify_api_reload():
    """Notify the API to reload the model after deployment."""
    import requests
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        api_host = os.getenv("API_HOST", "localhost")
        api_port = os.getenv("API_PORT", "8000")
        api_token = os.getenv("API_TOKEN")
        
        if not api_token:
            logger.warning("API_TOKEN not found - skipping API notification")
            return
        
        api_url = f"http://{api_host}:{api_port}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{api_url}/model/reload",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info("API notified successfully of model update")
            else:
                logger.warning(f"API reload failed: {result.get('message')}")
        else:
            logger.warning(f"API notification failed with status {response.status_code}")
            
    except Exception as e:
        logger.warning(f"Could not notify API of model update: {e}")

# Call API notification at the end of deployment
if __name__ == "__main__":
    # ... existing deployment code ...
    
    # Add this at the very end
    notify_api_reload()
'''
    
    # Check if API integration is already present
    if "notify_api_reload" not in deploy_content:
        # Add the integration code
        modified_content = deploy_content + api_integration_code
        
        # Write back to file
        with open(deploy_script, 'w') as f:
            f.write(modified_content)
        
        logger.info("API integration added to deploy_model.py")
    else:
        logger.info("API integration already present in deploy_model.py")
    
    return True

def create_api_test_script():
    """Create a test script for the API."""
    test_script_content = '''#!/usr/bin/env python3
"""
Test script for the ML Pipeline API.
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_api():
    """Test the API endpoints."""
    api_host = os.getenv("API_HOST", "localhost")
    api_port = os.getenv("API_PORT", "8000")
    api_token = os.getenv("API_TOKEN")
    
    base_url = f"http://{api_host}:{api_port}"
    
    print(f"Testing API at {base_url}")
    
    # Test health endpoint
    print("\\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test model info endpoint (without auth)
    print("\\n2. Testing model info endpoint (no auth)...")
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    if not api_token:
        print("\\nAPI_TOKEN not found - skipping authenticated tests")
        return
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Test model info endpoint (with auth)
    print("\\n3. Testing model info endpoint (with auth)...")
    try:
        response = requests.get(f"{base_url}/model/info", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test prediction endpoint
    print("\\n4. Testing prediction endpoint...")
    try:
        # Sample data for classification model
        test_data = {
            "data": {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0,
                "feature4": 4.0,
                "feature5": 5.0
            }
        }
        
        response = requests.post(
            f"{base_url}/predict",
            headers=headers,
            json=test_data
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test model reload endpoint
    print("\\n5. Testing model reload endpoint...")
    try:
        response = requests.post(f"{base_url}/model/reload", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
'''
    
    with open("test_api.py", 'w') as f:
        f.write(test_script_content)
    
    # Make it executable
    os.chmod("test_api.py", 0o755)
    
    logger.info("API test script created: test_api.py")

def create_docker_compose():
    """Create a docker-compose file for easy deployment."""
    docker_compose_content = '''version: '3.8'

services:
  ml-pipeline-api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - API_TOKEN=${API_TOKEN}
      - DEBUG=false
    volumes:
      - ./deploy:/app/deploy:ro
      - ./data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  model-monitor:
    build:
      context: .
      dockerfile: api/Dockerfile
    command: ["python", "api/reload_service.py"]
    environment:
      - API_HOST=ml-pipeline-api
      - API_PORT=8000
      - API_TOKEN=${API_TOKEN}
      - WATCH_PATH=/app/deploy/v_best
    volumes:
      - ./deploy:/app/deploy:ro
    depends_on:
      - ml-pipeline-api
    restart: unless-stopped
'''
    
    with open("docker-compose.yml", 'w') as f:
        f.write(docker_compose_content)
    
    logger.info("Docker Compose file created: docker-compose.yml")

def main():
    """Main integration function."""
    logger.info("Starting ML Pipeline API integration...")
    
    # Check if we're in the right directory
    if not Path("deploy").exists() or not Path("src").exists():
        logger.error("Not in ML pipeline project directory. Please run from project root.")
        return False
    
    try:
        # Integrate API with pipeline
        if not integrate_api_with_pipeline():
            logger.error("Failed to integrate API with pipeline")
            return False
        
        # Create test script
        create_api_test_script()
        
        # Create Docker Compose file
        create_docker_compose()
        
        # Create README for API
        readme_content = '''# ML Pipeline API

## Overview
This FastAPI REST API serves the v_best machine learning model from the ML pipeline.

## Features
- **Authentication**: Token-based authentication for secure access
- **Model Serving**: Serve predictions from the v_best model
- **Auto-reload**: Automatically reload model when v_best is updated
- **Health Checks**: Monitor API and model status
- **Documentation**: Automatic Swagger/OpenAPI documentation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r api/requirements.txt
```

### 2. Start the API
```bash
python start_api.py
```

### 3. Test the API
```bash
python test_api.py
```

## API Endpoints

### Health Check (Public)
```bash
GET /health
```

### Model Information (Optional Auth)
```bash
GET /model/info
```

### Make Predictions (Requires Auth)
```bash
POST /predict
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "data": {
    "feature1": 1.0,
    "feature2": 2.0,
    "feature3": 3.0
  }
}
```

### Reload Model (Requires Auth)
```bash
POST /model/reload
Authorization: Bearer YOUR_TOKEN
```

## Configuration

Edit `.env` file:
```
API_TOKEN=your-secret-token
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

## Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Integration with Pipeline

The API automatically integrates with your existing ML pipeline:
- Monitors `deploy/v_best/` directory for model updates
- Automatically reloads model when v_best is updated
- Provides endpoints for model information and predictions

## Documentation

Visit http://localhost:8000/docs for interactive API documentation.
'''
        
        with open("api/README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info("API README created")
        
        logger.info("✅ ML Pipeline API integration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start the API: python start_api.py")
        logger.info("2. Test the API: python test_api.py")
        logger.info("3. Visit documentation: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return False

if __name__ == "__main__":
    main()
'''