# Start with a lightweight version of Ubuntu
FROM ubuntu:22.04

# Install python3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Upgrade pip
RUN pip3 install --upgrade pip

# Install the required Python packages
RUN pip3 install fastapi-poe==0.0.23 \
    'fireworks-ai>=0.11.1' \
    boto3 \
    Pillow

WORKDIR /app

COPY ./ /app

RUN pip install .

# Expose the port that your FastAPI app will run on
EXPOSE 80
