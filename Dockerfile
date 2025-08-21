FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .


# "sh" use the shell # -c run cmd as string # Run Streamlit # server port with fallback (dynamic port) # 0.0.0.0 for making the app accessible from outside the container. # ${PORT} to run on GCP
CMD ["sh","-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]