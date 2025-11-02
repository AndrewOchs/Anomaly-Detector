"""
Custom exception classes for the application.
"""
from fastapi import HTTPException, status


class AnomalyDetectorException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(HTTPException):
    """Raised when authentication fails."""
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class UserNotFoundError(HTTPException):
    """Raised when a user is not found."""
    def __init__(self, detail: str = "User not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class UserAlreadyExistsError(HTTPException):
    """Raised when attempting to create a user that already exists."""
    def __init__(self, detail: str = "User already exists"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


class InvalidCredentialsError(HTTPException):
    """Raised when login credentials are invalid."""
    def __init__(self, detail: str = "Incorrect username or password"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class InvalidTokenError(HTTPException):
    """Raised when a token is invalid or expired."""
    def __init__(self, detail: str = "Invalid or expired token"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class InactiveUserError(HTTPException):
    """Raised when a user account is inactive."""
    def __init__(self, detail: str = "User account is inactive"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


class DatasetNotFoundError(HTTPException):
    """Raised when a dataset is not found."""
    def __init__(self, detail: str = "Dataset not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class FileValidationError(HTTPException):
    """Raised when file validation fails."""
    def __init__(self, detail: str = "File validation failed"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


class AnalysisError(HTTPException):
    """Raised when analysis execution fails."""
    def __init__(self, detail: str = "Analysis execution failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )
