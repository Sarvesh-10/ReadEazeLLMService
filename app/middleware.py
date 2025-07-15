from fastapi.responses import JSONResponse
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
                return JSONResponse({"error": "Authorization token not found"}, status_code=401)

            try:
                payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=['HS256'])
                
                logger.info(f"Decoded JWT payload:")
                request.state.user_id = payload['user_id']
                logger.info(f"User ID from JWT: {request.state.user_id}")
            except jwt.ExpiredSignatureError:
                return JSONResponse({"error": "Token has expired"}, status_code=401)
            except jwt.InvalidTokenError:
                return JSONResponse({"error": "Invalid token"}, status_code=401)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        response = await call_next(request)
        return response
