"""
Data validation utilities.
"""
import re
from typing import Optional


def is_valid_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_strong_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength.

    Requirements:
    - At least 8 characters
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"

    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing potentially dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path separators and keep only safe characters
    safe_name = re.sub(r'[^\w\s.-]', '', filename)
    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip('. ')
    # Replace multiple spaces with single space
    safe_name = re.sub(r'\s+', '_', safe_name)

    return safe_name if safe_name else "unnamed_file"


def is_allowed_file_extension(filename: str, allowed_extensions: list[str]) -> bool:
    """
    Check if file has an allowed extension.

    Args:
        filename: Filename to check
        allowed_extensions: List of allowed extensions (without dot)

    Returns:
        True if allowed, False otherwise
    """
    if '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()
    return ext in [e.lower().strip('.') for e in allowed_extensions]
