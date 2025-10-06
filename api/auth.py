"""
API Key 认证模块
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from .config import API_KEYS, ENABLE_API_KEY_AUTH


# HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> str:
    """
    验证 API Key
    
    Args:
        credentials: HTTP Authorization 头中的凭证
    
    Returns:
        验证通过的 API Key
    
    Raises:
        HTTPException: 认证失败时抛出 401 错误
    """
    # 如果禁用了认证，直接返回
    if not ENABLE_API_KEY_AUTH:
        return "disabled"
    
    # 检查是否提供了凭证
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "未提供 API Key。请在 Authorization 头中提供 Bearer token。",
                    "type": "invalid_request_error",
                    "code": "missing_api_key"
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证 API Key
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "无效的 API Key。",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key
