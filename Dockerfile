FROM python:latest

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"
ENV USERNAME="myuser"

WORKDIR /mainFlow
COPY requirements.txt /mainFlow/
#RUN pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /mainFlow


CMD [ "python3", "radAImetaflow.py", "run"]
