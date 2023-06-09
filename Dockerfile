FROM python:3.10
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
RUN pytest test_model.py
CMD streamlit run model.py
