# -------------------------------
# Dockerfile for Streamlit App
# -------------------------------

# Use a lightweight official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files to /app in the container
COPY . .

# Install dependencies
# You can either use requirements.txt or inline install
RUN pip install --no-cache-dir streamlit pandas openpyxl

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
