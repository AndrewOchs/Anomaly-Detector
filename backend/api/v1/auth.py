"""
Authentication API endpoints.
"""
from datetime import timedelta
from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from config.database import get_db
from config.settings import settings
from backend.dependencies import CurrentActiveUser, DatabaseSession
from backend.models.user import User
from backend.schemas.auth import UserRegister, UserLogin, Token, TokenRefresh
from backend.schemas.user import UserResponse
from backend.services.auth_service import auth_service
from backend.utils.exceptions import (
    UserAlreadyExistsError,
    InvalidCredentialsError,
    InvalidTokenError,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserRegister, db: DatabaseSession):
    """
    Register a new user.

    - **email**: Valid email address
    - **username**: Unique username (3-50 characters)
    - **password**: Password (min 8 characters)

    Returns the created user information (without password).
    """
    logger.info(f"Registration attempt for username: {user_data.username}")

    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()

    if existing_user:
        if existing_user.email == user_data.email:
            logger.warning(f"Registration failed: email {user_data.email} already exists")
            raise UserAlreadyExistsError("Email already registered")
        else:
            logger.warning(f"Registration failed: username {user_data.username} already exists")
            raise UserAlreadyExistsError("Username already taken")

    # Create new user
    hashed_password = auth_service.hash_password(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        is_active=True
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"User registered successfully: {new_user.username} (ID: {new_user.id})")
    return new_user


@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: DatabaseSession):
    """
    Login with username/email and password.

    - **username**: Username or email address
    - **password**: User password

    Returns JWT access and refresh tokens.
    """
    logger.info(f"Login attempt for: {user_data.username}")

    # Authenticate user
    user = auth_service.authenticate_user(db, user_data.username, user_data.password)

    # Create tokens
    access_token = auth_service.create_access_token(
        data={"sub": user.id, "username": user.username}
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": user.id, "username": user.username}
    )

    logger.info(f"User logged in successfully: {user.username}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login/form", response_model=Token)
def login_form(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Login using OAuth2 password flow (for Swagger UI compatibility).

    This endpoint is identical to /login but uses form data instead of JSON.
    Required for OAuth2PasswordBearer scheme.
    """
    logger.info(f"Form login attempt for: {form_data.username}")

    # Authenticate user
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)

    # Create tokens
    access_token = auth_service.create_access_token(
        data={"sub": user.id, "username": user.username}
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": user.id, "username": user.username}
    )

    logger.info(f"User logged in successfully (form): {user.username}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
def refresh_access_token(token_data: TokenRefresh, db: DatabaseSession):
    """
    Refresh access token using a valid refresh token.

    - **refresh_token**: Valid JWT refresh token

    Returns a new access token.
    """
    try:
        # Decode refresh token
        token_payload = auth_service.decode_token(token_data.refresh_token)

        # Verify user still exists and is active
        user = db.query(User).filter(User.id == token_payload.user_id).first()
        if not user or not user.is_active:
            raise InvalidTokenError("Invalid refresh token")

        # Create new access token
        access_token = auth_service.create_access_token(
            data={"sub": user.id, "username": user.username}
        )

        logger.info(f"Access token refreshed for user: {user.username}")

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        raise InvalidTokenError("Could not refresh token")


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: CurrentActiveUser):
    """
    Get current authenticated user information.

    Requires valid JWT token in Authorization header.
    """
    logger.info(f"User info requested: {current_user.username}")
    return current_user
