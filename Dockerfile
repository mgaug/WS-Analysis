FROM python:3.9 
COPY . /ws

WORKDIR /ws
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

CMD ["python","run.py"]
#RUN tail -f /dev/null
