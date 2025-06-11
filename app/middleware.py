from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException
import jwt
import os
import logging

class JWTAuthMiddleware(BaseHTTPMiddleware):
    '''Middleware for JWT Authentication'''

    async def dispatch(self, request, call_next):
        logging.info(f"Processing request: {request.method} {request.url.path}")
        if request.url.path.startswith('/api'):
            logging.info("Request is for API endpoint, checking JWT token...")
            # Try reading the token from the cookie
            print(request.cookies.items())
            token = request.cookies.get('token')
            logging.info(f"Received token: {token}")  # Log the received token

            if not token:
                raise HTTPException(status_code=401, detail='Authorization token missing')

            try:
                payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=['HS256'])
                
                logging.info(f"Decoded JWT payload: {payload}")
                request.state.user_id = payload['user_id']
                logging.info(f"User ID from token: {request.state.user_id}")
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail='Token expired')
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail='Invalid token')
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))  

        response = await call_next(request)
        return response
