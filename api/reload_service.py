"""
Service to monitor v_best directory and reload the API when model is updated.
"""
import os
import time
import logging
import requests
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelUpdateHandler(FileSystemEventHandler):
    """Handler for model file changes."""
    
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token
        self.last_reload_time = 0
        self.reload_cooldown = 5  # seconds
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Check if it's a model-related file
        if any(filename in event.src_path for filename in ['model.pkl', 'model_metadata.json', 'v_best_metadata.json']):
            current_time = time.time()
            
            # Avoid rapid successive reloads
            if current_time - self.last_reload_time > self.reload_cooldown:
                logger.info(f"Model file changed: {event.src_path}")
                self.reload_model()
                self.last_reload_time = current_time
    
    def reload_model(self):
        """Send reload request to the API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.api_url}/model/reload",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("Model reloaded successfully via API")
                else:
                    logger.error(f"Model reload failed: {result.get('message')}")
            else:
                logger.error(f"API reload request failed with status {response.status_code}")
                
        except requests.RequestException as e:
            logger.error(f"Error sending reload request to API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during model reload: {e}")

def monitor_v_best_directory(watch_path: str, api_url: str, api_token: str):
    """
    Monitor the v_best directory for changes and reload the model when updated.
    
    Args:
        watch_path: Path to the v_best directory to monitor
        api_url: URL of the FastAPI service
        api_token: Authentication token for the API
    """
    if not Path(watch_path).exists():
        logger.error(f"Watch path does not exist: {watch_path}")
        return
    
    logger.info(f"Starting to monitor directory: {watch_path}")
    logger.info(f"API URL: {api_url}")
    
    event_handler = ModelUpdateHandler(api_url, api_token)
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=True)
    
    try:
        observer.start()
        logger.info("Model monitoring service started")
        
        # Keep the service running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping model monitoring service...")
        observer.stop()
    
    observer.join()
    logger.info("Model monitoring service stopped")

def check_api_health(api_url: str, max_retries: int = 5, retry_delay: int = 5):
    """
    Check if the API is running and healthy.
    
    Args:
        api_url: URL of the FastAPI service
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        bool: True if API is healthy, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "ok":
                    logger.info("API is healthy and ready")
                    return True
                else:
                    logger.warning(f"API returned non-OK status: {health_data}")
            else:
                logger.warning(f"API health check failed with status {response.status_code}")
                
        except requests.RequestException as e:
            logger.warning(f"API health check attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error("API health check failed after all retries")
    return False

def main():
    """Main function to start the monitoring service."""
    # Configuration - Fix Windows networking issue
    watch_path = os.getenv("WATCH_PATH", "deploy/v_best")
    api_host = os.getenv("API_HOST", "localhost")  # Changed from 0.0.0.0 to localhost
    api_port = os.getenv("API_PORT", "8000")
    api_token = os.getenv("API_TOKEN")
    
    if not api_token:
        logger.error("API_TOKEN not found in environment variables")
        return
    
    # Use localhost instead of 0.0.0.0 for Windows compatibility
    if api_host == "0.0.0.0":
        api_host = "localhost"
    
    api_url = f"http://{api_host}:{api_port}"
    
    # Convert relative path to absolute
    if not os.path.isabs(watch_path):
        watch_path = os.path.abspath(watch_path)
    
    logger.info("ML Pipeline Model Monitor Service")
    logger.info(f"Watch path: {watch_path}")
    logger.info(f"API URL: {api_url}")
    
    # Check if API is running
    if not check_api_health(api_url):
        logger.error("Cannot start monitoring - API is not healthy")
        return
    
    # Start monitoring
    try:
        monitor_v_best_directory(watch_path, api_url, api_token)
    except Exception as e:
        logger.error(f"Monitoring service failed: {e}")

if __name__ == "__main__":
    main()