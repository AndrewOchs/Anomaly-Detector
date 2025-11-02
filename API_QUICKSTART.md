# API Quick Start Guide

This guide will help you quickly test the FastAPI backend authentication system.

## Prerequisites

- PostgreSQL installed and running
- Database initialized (see SETUP.md)
- Python virtual environment activated
- Dependencies installed (`pip install -r requirements.txt`)

## Step 1: Create .env File

If you haven't already, create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and update at least these values:
```bash
# Generate a secure secret key:
# python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=your-generated-secret-key-here

# Database (if different from defaults)
DATABASE_URL=postgresql://anomaly_user:anomaly_password@localhost:5432/anomaly_detector
```

## Step 2: Start the API Server

### Option 1: Using the run script
```bash
python scripts/run_api.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn backend.main:app --reload --port 8000
```

### Option 3: Using the main module
```bash
python -m backend.main
```

The API will be available at:
- **API Endpoints**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## Step 3: Test the API

### Method 1: Using the Interactive Docs (Recommended)

1. Open http://localhost:8000/docs in your browser
2. You'll see all available endpoints with interactive testing capability

#### Register a New User:
1. Expand the `POST /api/v1/auth/register` endpoint
2. Click "Try it out"
3. Enter user details:
   ```json
   {
     "email": "test@example.com",
     "username": "testuser",
     "password": "testpass123"
   }
   ```
4. Click "Execute"
5. You should receive a 201 response with user data

#### Login:
1. Expand the `POST /api/v1/auth/login` endpoint
2. Click "Try it out"
3. Enter credentials:
   ```json
   {
     "username": "testuser",
     "password": "testpass123"
   }
   ```
4. Click "Execute"
5. Copy the `access_token` from the response

#### Test Protected Endpoint:
1. Click the "Authorize" button at the top right
2. Enter: `Bearer <your-access-token>` (replace with your token)
3. Click "Authorize" then "Close"
4. Expand the `GET /api/v1/auth/me` endpoint
5. Click "Try it out" then "Execute"
6. You should see your user information

### Method 2: Using curl

#### Register:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "testpass123"
  }'
```

#### Login:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```

Save the `access_token` from the response.

#### Get User Info (Protected Route):
```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer <your-access-token>"
```

### Method 3: Using Python requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Register
response = requests.post(
    f"{BASE_URL}/api/v1/auth/register",
    json={
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123"
    }
)
print("Register:", response.json())

# Login
response = requests.post(
    f"{BASE_URL}/api/v1/auth/login",
    json={
        "username": "testuser",
        "password": "testpass123"
    }
)
tokens = response.json()
access_token = tokens["access_token"]
print("Login:", tokens)

# Get user info (protected)
response = requests.get(
    f"{BASE_URL}/api/v1/auth/me",
    headers={"Authorization": f"Bearer {access_token}"}
)
print("User Info:", response.json())
```

## Available Endpoints

### Public Endpoints (No Authentication Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API root information |
| GET | `/health` | Health check and status |
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login and get tokens |
| POST | `/api/v1/auth/login/form` | Login (OAuth2 form) |
| POST | `/api/v1/auth/refresh` | Refresh access token |

### Protected Endpoints (Require Authentication)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/auth/me` | Get current user info |

## Response Examples

### Successful Registration (201)
```json
{
  "id": 1,
  "email": "test@example.com",
  "username": "testuser",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

### Successful Login (200)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### User Info (200)
```json
{
  "id": 1,
  "email": "test@example.com",
  "username": "testuser",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

## Error Responses

### User Already Exists (400)
```json
{
  "detail": "Email already registered"
}
```

### Invalid Credentials (401)
```json
{
  "detail": "Incorrect username or password"
}
```

### Unauthorized Access (401)
```json
{
  "detail": "Could not validate credentials"
}
```

### Validation Error (422)
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "value is not a valid email address",
      "type": "value_error.email"
    }
  ]
}
```

## Troubleshooting

### Server won't start
- Check PostgreSQL is running: `psql -U anomaly_user -d anomaly_detector`
- Verify `.env` file exists and has correct DATABASE_URL
- Check port 8000 is not already in use: `lsof -i :8000`

### Authentication fails
- Verify SECRET_KEY is set in `.env`
- Check user exists in database
- Ensure token hasn't expired (default: 30 minutes)

### Database errors
- Run database initialization: `python scripts/init_db.py`
- Check database connection in logs
- Verify PostgreSQL credentials

## Next Steps

Once the authentication is working:
1. Test all endpoints in the interactive docs
2. Proceed to Phase 3: File upload functionality
3. Integrate with the Dash frontend

## Security Notes

For production deployment:
- Use strong, unique SECRET_KEY
- Enable HTTPS/TLS
- Set appropriate CORS_ORIGINS
- Use environment-specific .env files
- Never commit .env to version control
- Implement rate limiting
- Use secure password requirements
