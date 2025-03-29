from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException
import jwt
import os

class JWTAuthMiddleware(BaseHTTPMiddleware):
    '''Middleware for JWT Authentication'''

    async def dispatch(self, request, call_next):
        print("HERE in middleware")
        if request.url.path.startswith('/api'):
            print("HERE")
            # Try reading the token from the cookie
            print(request.cookies.items())
            token = request.cookies.get('token')
            print("HERE", token)

            if not token:
                raise HTTPException(status_code=401, detail='Authorization token missing')

            try:
                payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=['HS256'])
                print("payload is ",payload)
                request.state.user_id = payload['user_id']
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail='Token expired')
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail='Invalid token')
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))  

        response = await call_next(request)
        return response
