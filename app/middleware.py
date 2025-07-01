from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException
import jwt
import os
from .customLogging import logger

class JWTAuthMiddleware(BaseHTTPMiddleware):
    '''Middleware for JWT Authentication'''

    async def dispatch(self, request, call_next):
        logger.info(f"Processing request: {request.method} {request.url.path}")
        # Always check JWT (unless you want to skip auth for public routes like / or /test-cookie)
        protected_paths = ["/chat", "/image", "/something-else"]

        if any(request.url.path.startswith(p) for p in protected_paths):
            logger.info(f"Checking JWT for path: {request.url.path}")
            # Try reading the token from the cookie
            print(request.cookies.items())
            token = request.cookies.get('token')
            

            if not token:
                raise HTTPException(status_code=401, detail='Authorization token missing')

            try:
                payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=['HS256'])
                
                logger.info(f"Decoded JWT payload:")
                request.state.user_id = payload['user_id']
                logger.info(f"User ID from JWT: {request.state.user_id}")
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail='Token expired')
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail='Invalid token')
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))  

        response = await call_next(request)
        return response
