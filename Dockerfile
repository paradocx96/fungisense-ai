FROM python:3-slim

# setup okteto message
COPY bashrc /root/.bashrc

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY model.h5 model.h5
COPY mushrooms.csv mushrooms.csv
COPY app.py app.py

EXPOSE 8080

CMD ["python", "app.py" ]
