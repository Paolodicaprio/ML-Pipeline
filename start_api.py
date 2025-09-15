#!/usr/bin/env python3
"""
Script to start the ML Pipeline API server.
"""
import os
import sys
import subprocess
import logging
import signal
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIServer:
    """Manages the FastAPI server process."""
    
    def __init__(self):
        self.process = None
        self.monitor_process = None
        self.shutdown_event = threading.Event()
    
    def start_api(self):
        """Start the FastAPI server."""
        try:
            # Check if model files exist
            model_path = Path("deploy/v_best/model.pkl")
            if not model_path.exists():
                logger.warning("v_best model not found. API will start but predictions will fail until model is available.")
            
            # Get configuration from environment
            host = os.getenv("API_HOST", "0.0.0.0")
            port = int(os.getenv("API_PORT", 8000))
            debug = os.getenv("DEBUG", "True").lower() == "true"
            
            # Start the API server
            cmd = [
                sys.executable, "-m", "uvicorn",
                "api.main:app",
                "--host", host,
                "--port", str(port),
                "--log-level", "info"
            ]
            
            if debug:
                cmd.append("--reload")
            
            logger.info(f"Starting API server on {host}:{port}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(cmd)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_monitor(self):
        """Start the model monitoring service."""
        try:
            # Set environment variable for monitoring service to use localhost
            env = os.environ.copy()
            if env.get("API_HOST") == "0.0.0.0":
                env["API_HOST"] = "localhost"
            
            cmd = [sys.executable, "api/reload_service.py"]
            
            logger.info("Starting model monitoring service...")
            
            self.monitor_process = subprocess.Popen(cmd, env=env)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring service: {e}")
            return False
    
    def stop(self):
        """Stop all services."""
        logger.info("Stopping services...")
        
        self.shutdown_event.set()
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        if self.monitor_process:
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
        
        logger.info("All services stopped")

def signal_handler(signum, frame, server):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    server.stop()
    sys.exit(0)

def main():
    """Main function to start the API and monitoring services."""
    logger.info("ML Pipeline API Server")
    
    # Check if we're in the right directory
    if not Path("deploy").exists():
        logger.error("Not in ML pipeline project directory. Please run from project root.")
        sys.exit(1)
    
    # Install API dependencies
    logger.info("Installing API dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "api/requirements.txt"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)
    
    # Create server instance
    server = APIServer()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, server))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, server))
    
    try:
        # Start API server
        if not server.start_api():
            logger.error("Failed to start API server")
            sys.exit(1)
        
        # Wait a bit for API to start
        time.sleep(5)  # Increased wait time
        
        # Start monitoring service
        if not server.start_monitor():
            logger.warning("Failed to start monitoring service - continuing without it")
        
        logger.info("All services started successfully")
        logger.info("API documentation available at: http://localhost:8000/docs")
        logger.info("Health check available at: http://localhost:8000/health")
        logger.info("Press Ctrl+C to stop all services")
        
        # Keep the main process running
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if server.process and server.process.poll() is not None:
                    logger.error("API server process died")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
    except Exception as e:
        logger.error(f"Error running services: {e}")
    finally:
        server.stop()

if __name__ == "__main__":
    main()