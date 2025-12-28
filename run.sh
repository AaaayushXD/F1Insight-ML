#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run FastAPI application with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

