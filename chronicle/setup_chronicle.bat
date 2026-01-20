@echo off
REM Quick setup script for Google Chronicle SIEM Integration

echo ========================================
echo Google Chronicle SIEM Setup
echo ========================================
echo.

REM Check Python
echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo Installing Chronicle dependencies...
pip install -q google-auth google-auth-httplib2 requests fastapi uvicorn

echo.
echo ========================================
echo Configuration Setup
echo ========================================
echo.

REM Check if config exists
if exist chronicle_config.json (
    echo Found existing chronicle_config.json
    set /p OVERWRITE="Overwrite existing config? (y/n): "
    if /i not "%OVERWRITE%"=="y" goto :skip_config
)

REM Copy template
echo Creating chronicle_config.json from template...
copy chronicle_config_template.json chronicle_config.json

echo.
echo Please edit chronicle_config.json and fill in:
echo   1. customer_id: Your Chronicle customer ID (C00000000)
echo   2. region: Your Chronicle region (us/europe/asia)
echo   3. credentials_file: Path to your service account JSON
echo.

:skip_config

REM Check for credentials file
if exist chronicle_credentials.json (
    echo ✓ Found chronicle_credentials.json
) else (
    echo.
    echo ========================================
    echo Service Account Setup Required
    echo ========================================
    echo.
    echo You need to create a Google Cloud service account:
    echo.
    echo 1. Go to Google Cloud Console
    echo 2. Create service account: chronicle-anomaly-detector
    echo 3. Grant role: Chronicle Agent
    echo 4. Create JSON key
    echo 5. Download as chronicle_credentials.json
    echo.
    echo See GOOGLE_SIEM_SETUP.md for detailed instructions.
    echo.
)

echo.
echo ========================================
echo Testing Configuration
echo ========================================
echo.

if exist chronicle_credentials.json (
    echo Running connection test...
    python google_chronicle_integration.py --test

    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✓ Chronicle integration configured successfully!
        echo.
        echo You can now:
        echo   - Run API with Chronicle: python anomaly_api_chronicle.py
        echo   - Create detection rules: python google_chronicle_integration.py --create-rules
        echo   - Test integration: python test_api.py
    ) else (
        echo.
        echo ✗ Connection test failed
        echo Please check your credentials and configuration
    )
) else (
    echo Skipping test (credentials file not found)
    echo.
    echo Next steps:
    echo   1. Place chronicle_credentials.json in this directory
    echo   2. Edit chronicle_config.json with your customer ID
    echo   3. Run: python google_chronicle_integration.py --test
)

echo.
echo ========================================
echo Setup Complete
echo ========================================
echo.
echo For detailed instructions, see:
echo   GOOGLE_SIEM_SETUP.md
echo.

pause
