FROM python:3.11

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . .
EXPOSE 5046 8501 

CMD streamlit run app.py