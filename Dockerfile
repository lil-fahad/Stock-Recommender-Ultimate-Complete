
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip &&     pip install -r requirements-upgraded.txt

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
