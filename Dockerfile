FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m paytrace && chown -R paytrace /app
USER paytrace
EXPOSE 8501 8000
# the web app by default; the admin analytics dashboard: docker run ... ./run_v2.sh
ENV HOST=0.0.0.0
CMD ["sh", "-c", "./run_app.sh"]
