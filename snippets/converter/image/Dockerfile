# ==============================================================================
# Copyright 2021 Google LLC All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

FROM python:3.6
# COPY requirements.txt /app/requirements.txt
RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && apt-get update && apt install edgetpu-compiler=15.0 -y 
# RUN apt-cache showpkg edgetpu
RUN pip install --upgrade pip
RUN pip install fastapi==0.41.0 pydantic==0.32.2 Pillow==6.2.0 starlette==0.12.9 six==1.12.0 uvicorn==0.9.0 promise==2.2.1 httptools==0.0.13 gunicorn==19.9.0 python-multipart==0.0.5 aiofiles==0.4.0
RUN pip install tensorflowjs==2.0.1
RUN pip install scipy==1.4.1
RUN pip install tensorflow==2.3.0

WORKDIR /app
COPY api.py ./
CMD exec gunicorn --bind :8080  -k uvicorn.workers.UvicornWorker --workers 1 --threads 8 --timeout 300 --reload  api:app