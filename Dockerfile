# Start with a lightweight version of Ubuntu
FROM ubuntu:22.04

# Install python3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Install the required Python packages
RUN pip3 install fastapi-poe==0.0.23 \
    'fireworks-ai>=0.8.0' \
    boto3 \
    Pillow

WORKDIR /app

COPY ./ /app

RUN pip install .

# Expose the port that your FastAPI app will run on
EXPOSE 80