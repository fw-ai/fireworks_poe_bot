set -ex
docker build --platform linux/amd64 -t fireworks-poe-bot:latest -f ./Dockerfile .
