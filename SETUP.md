# Production Setup Guide

This guide will walk you through setting up the Anomaly Detection Platform for local development and production deployment.

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher
- pip (Python package manager)
- virtualenv or venv

## Step 1: PostgreSQL Installation & Setup

### On macOS (using Homebrew):
```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Create database user and database
psql postgres
```

### On Ubuntu/Debian:
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Switch to postgres user
sudo -u postgres psql
```

### On Windows:
1. Download PostgreSQL installer from https://www.postgresql.org/download/windows/
2. Run the installer and follow the setup wizard
3. Remember the password you set for the postgres user
4. Open pgAdmin or use the command line tools

### Create Database and User:

In the PostgreSQL prompt (`psql`):
```sql
-- Create database user
CREATE USER anomaly_user WITH PASSWORD 'anomaly_password';

-- Create database
CREATE DATABASE anomaly_detector OWNER anomaly_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE anomaly_detector TO anomaly_user;

-- Exit psql
\q
```

## Step 2: Python Environment Setup

```bash
# Clone the repository (if not already done)
cd Anomaly-Detector

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Environment Configuration

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env file with your settings
nano .env  # or use your preferred text editor
```

### Required `.env` Configuration:

```bash
# Update these values in your .env file:

# Database (match what you created in PostgreSQL)
DATABASE_URL=postgresql://anomaly_user:anomaly_password@localhost:5432/anomaly_detector

# Security - IMPORTANT: Change this to a random string in production
SECRET_KEY=your-secure-random-secret-key-here

# API Keys (get these from respective services)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
```

### Generate a Secure SECRET_KEY:

```python
# Run this in Python to generate a secure key:
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Step 4: Get API Keys (Optional for Live Data)

### Alpha Vantage API (Stock Market Data):
1. Visit: https://www.alphavantage.co/support/#api-key
2. Sign up for a free API key
3. Add to `.env`: `ALPHA_VANTAGE_API_KEY=your-key-here`

### Other Optional APIs:
- **OpenWeather**: https://openweathermap.org/api
- **CoinGecko**: https://www.coingecko.com/en/api

## Step 5: Database Initialization

```bash
# Initialize the database and create tables
python scripts/init_db.py
```

You should see output like:
```
============================================================
Database Initialization Script
============================================================
Environment: development
Database URL: postgresql://anomaly_user:***@localhost:5432/anomaly_detector
============================================================
Checking database connection...
Database connection successful
Creating database tables...
Successfully created 4 tables:
  ✓ users
  ✓ datasets
  ✓ analysis_results
  ✓ user_sessions
============================================================
Database initialization completed successfully!
============================================================
```

## Step 6: Run Database Migrations (Alternative Method)

If you prefer using Alembic migrations:

```bash
# Run migrations to create tables
alembic upgrade head

# To create a new migration after model changes:
alembic revision --autogenerate -m "Description of changes"

# Apply new migrations:
alembic upgrade head

# Rollback one migration:
alembic downgrade -1
```

## Step 7: Verify Installation

Test database connection:
```python
python -c "from config.database import check_db_connection; print('Success!' if check_db_connection() else 'Failed')"
```

## Common Issues & Troubleshooting

### Issue: "connection refused" or "could not connect to server"
**Solution:** Ensure PostgreSQL is running:
```bash
# macOS:
brew services list
brew services restart postgresql@15

# Linux:
sudo systemctl status postgresql
sudo systemctl restart postgresql
```

### Issue: "password authentication failed"
**Solution:**
1. Check your `.env` file has correct credentials
2. Verify PostgreSQL user exists: `psql -U anomaly_user -d anomaly_detector`
3. Reset password if needed:
   ```sql
   ALTER USER anomaly_user WITH PASSWORD 'new_password';
   ```

### Issue: "database does not exist"
**Solution:** Create the database:
```bash
psql -U postgres
CREATE DATABASE anomaly_detector OWNER anomaly_user;
```

### Issue: Import errors
**Solution:** Ensure you're in the project root and virtual environment is activated:
```bash
pwd  # Should show .../Anomaly-Detector
which python  # Should show venv path
pip list  # Verify packages are installed
```

## Next Steps

After successful setup:

1. **Start the FastAPI backend** (when implemented):
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

2. **Start the Dash frontend** (when implemented):
   ```bash
   python frontend/app.py
   ```

3. **Access the application**:
   - API: http://localhost:8000
   - Dashboard: http://localhost:8050
   - API Docs: http://localhost:8000/docs

## Production Deployment Considerations

1. **Environment Variables:**
   - Never commit `.env` to version control
   - Use strong, unique SECRET_KEY
   - Set `DEBUG=False` in production

2. **Database:**
   - Use managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)
   - Enable SSL connections
   - Regular backups

3. **Security:**
   - Use HTTPS (SSL/TLS certificates)
   - Configure CORS properly
   - Implement rate limiting
   - Regular security updates

4. **Deployment Options:**
   - Docker + Docker Compose
   - Heroku
   - AWS (EC2 + RDS)
   - Google Cloud Platform
   - DigitalOcean

## Support

For issues or questions:
- Check the main README.md
- Review error logs in `logs/app.log`
- Ensure all prerequisites are installed
