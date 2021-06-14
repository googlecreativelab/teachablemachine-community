FROM python:3.6

RUN pip install --upgrade pip
RUN pip install fastapi==0.41.0 pydantic==0.32.2 Pillow==6.2.0 starlette==0.12.9 six==1.12.0 uvicorn==0.9.0 promise==2.2.1 httptools==0.0.13 gunicorn==20.0.4 python-multipart==0.0.5 aiofiles==0.4.0
RUN pip install tensorflowjs==1.3.1
RUN pip install Pillow
RUN pip install tensorflow==1.15.0
RUN apt-get update
RUN apt-get install xxd

WORKDIR /app
COPY api.py ./
COPY sketch_templates ./
## appengine custom runtime must bind to 8080
CMD exec gunicorn --bind :8080  -k uvicorn.workers.UvicornWorker --workers 1 --threads 8 --timeout 300 --reload  api:app
