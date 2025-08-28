# Hospital Readmission Prediction App - Deployment Guide

## Issues Fixed

### 1. Port Configuration Issue
**Problem**: The app was configured to run on port 5000, but health checks expected port 8501.

**Solution**: Updated `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"  # Changed from "localhost" to allow external access
port = 8501          # Changed from 5000 to default Streamlit port
```

### 2. Missing Dependencies
**Problem**: SHAP library was missing from dependencies.

**Solution**: Added `shap>=0.41.0` to both `pyproject.toml` and `requirements.txt`.

## Deployment Options

### Option 1: Using requirements.txt
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: Using the startup script
```bash
# Install dependencies
pip install -r requirements.txt

# Run using the startup script
python run_streamlit.py
```

### Option 3: Direct Streamlit command
```bash
# Install dependencies
pip install -r requirements.txt

# Run with explicit configuration
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

## Environment Variables

For production deployment, you can set these environment variables:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

## Health Check

The app now includes a health check function. You can verify the app is running by:

1. **Browser**: Navigate to `http://localhost:8501`
2. **Health Check**: The app will respond with a health status in the session state

## Troubleshooting

### Connection Refused Error
If you still get connection refused errors:

1. **Check if the app is running**:
   ```bash
   ps aux | grep streamlit
   ```

2. **Check if port 8501 is in use**:
   ```bash
   netstat -tulpn | grep 8501
   ```

3. **Kill any existing processes**:
   ```bash
   pkill -f streamlit
   ```

4. **Restart the app**:
   ```bash
   streamlit run app.py
   ```

### Missing Dependencies
If you get import errors:

1. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

2. **Check Python version** (requires Python 3.11+):
   ```bash
   python --version
   ```

### Permission Issues
If you get permission errors:

1. **Make the startup script executable**:
   ```bash
   chmod +x run_streamlit.py
   ```

2. **Run with proper permissions**:
   ```bash
   sudo python run_streamlit.py
   ```

## Production Deployment

For production deployment on platforms like Streamlit Cloud, Railway, or Heroku:

1. **Use requirements.txt** for dependency management
2. **Set environment variables** for configuration
3. **Use the startup script** for consistent deployment
4. **Monitor logs** for any errors

## App Features

The app includes:
- Data loading and preprocessing
- Model training and evaluation
- Model explainability with SHAP
- Clinical decision support
- Performance dashboard
- Mobile-responsive design

## Support

If you encounter issues:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure the correct port is being used
4. Check firewall settings if deploying externally
