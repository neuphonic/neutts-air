"""
API Key Authentication Module
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from .config import API_KEYS, ENABLE_API_KEY_AUTH


# HTTP Bearer authentication scheme
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> str:
    """
    Verify API Key
    
    Args:
        credentials: Credentials from HTTP Authorization header
    
    Returns:
        Verified API Key
    
    Raises:
        HTTPException: Raises 401 error on authentication failure
    """
    # If authentication is disabled, return directly
    if not ENABLE_API_KEY_AUTH:
        return "disabled"
    
    # Check if credentials are provided
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "No API Key provided. Please provide a Bearer token in the Authorization header.",
                    "type": "invalid_request_error",
                    "code": "missing_api_key"
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify API Key
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API Key.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key
