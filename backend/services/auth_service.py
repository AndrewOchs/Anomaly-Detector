"""
Authentication service for handling JWT tokens and password hashing.
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config.settings import settings
from backend.models.user import User
from backend.schemas.auth import TokenData
from backend.utils.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    InvalidCredentialsError,
    InactiveUserError,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for authentication operations."""

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a plain text password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain text password against a hashed password.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.

        Args:
            data: Data to encode in the token
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })

        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """
        Create a JWT refresh token.

        Args:
            data: Data to encode in the token

        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })

        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> TokenData:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token to decode

        Returns:
            TokenData with user information

        Raises:
            InvalidTokenError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: int = payload.get("sub")
            username: str = payload.get("username")

            if user_id is None:
                logger.warning("Token missing user_id (sub) claim")
                raise InvalidTokenError("Invalid token payload")

            return TokenData(user_id=user_id, username=username)

        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            raise InvalidTokenError("Could not validate token")

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> User:
        """
        Authenticate a user with username/email and password.

        Args:
            db: Database session
            username: Username or email
            password: Plain text password

        Returns:
            User object if authentication successful

        Raises:
            InvalidCredentialsError: If credentials are invalid
            InactiveUserError: If user account is inactive
        """
        # Try to find user by username or email
        user = db.query(User).filter(
            (User.username == username) | (User.email == username)
        ).first()

        if not user:
            logger.info(f"Authentication failed: user '{username}' not found")
            raise InvalidCredentialsError()

        if not AuthService.verify_password(password, user.hashed_password):
            logger.info(f"Authentication failed: invalid password for user '{username}'")
            raise InvalidCredentialsError()

        if not user.is_active:
            logger.warning(f"Authentication failed: user '{username}' is inactive")
            raise InactiveUserError()

        logger.info(f"User '{username}' authenticated successfully")
        return user

    @staticmethod
    def get_current_user(db: Session, token: str) -> User:
        """
        Get current user from JWT token.

        Args:
            db: Database session
            token: JWT access token

        Returns:
            User object

        Raises:
            AuthenticationError: If token is invalid or user not found
            InactiveUserError: If user account is inactive
        """
        try:
            token_data = AuthService.decode_token(token)

            user = db.query(User).filter(User.id == token_data.user_id).first()

            if user is None:
                logger.warning(f"User ID {token_data.user_id} from token not found in database")
                raise AuthenticationError("User not found")

            if not user.is_active:
                logger.warning(f"Inactive user {user.username} attempted to access resource")
                raise InactiveUserError()

            return user

        except InvalidTokenError as e:
            raise AuthenticationError(str(e))


# Create singleton instance
auth_service = AuthService()
