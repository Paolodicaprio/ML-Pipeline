"""
Authentication middleware for the ML Pipeline API.
"""
import os
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Security scheme
security = HTTPBearer()

def get_api_token() -> str:
    """Get the API token from environment variables."""
    token = os.getenv("API_TOKEN")
    if not token:
        raise ValueError("API_TOKEN not found in environment variables")
    return token

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify the provided token against the configured API token.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        str: The verified token
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    expected_token = get_api_token()
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials

# Optional authentication for some endpoints
def verify_token_optional(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> Optional[str]:
    """
    Optionally verify token - used for endpoints that can work with or without auth.
    
    Args:
        credentials: Optional HTTP Authorization credentials
        
    Returns:
        Optional[str]: The verified token if provided and valid, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return verify_token(credentials)
    except HTTPException:
        return None