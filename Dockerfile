FROM python:3.9 
COPY . /ws

WORKDIR /ws
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \

CMD tail -f /dev/null
