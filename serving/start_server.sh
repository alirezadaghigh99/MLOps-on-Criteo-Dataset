
set -u

if ! [ -x "$(command -v docker)" ]; then
    echo "ERROR: This script requires Docker"
    exit 1
fi

# The name of the image to run is specified by the constant below
DOCKER_IMAGE_NAME=tensorflow/serving

echo Pulling the Docker image: $DOCKER_IMAGE_NAME

docker pull $DOCKER_IMAGE_NAME

# Dir for model exported for serving
LOCAL_MODEL_DIR=$1
# Make sure the trained model is available
if ! [ -d "$LOCAL_MODEL_DIR" ]; then
  echo "ERROR: Could not find the exported model directory $LOCAL_MODEL_DIR"
  exit 1
fi

echo Starting the Model Server to serve from: $LOCAL_MODEL_DIR

# Container model dir
CONTAINER_MODEL_DIR=/models/chicago_taxi

# Local port where to send inference requests
HOST_PORT=9000

# Where our container is listening
CONTAINER_PORT=8501

echo Model directory: $LOCAL_MODEL_DIR

docker run -it\
  -p 127.0.0.1:$HOST_PORT:$CONTAINER_PORT \
  -v $LOCAL_MODEL_DIR:$CONTAINER_MODEL_DIR \
  -e MODEL_NAME=chicago_taxi \
  --rm $DOCKER_IMAGE_NAME
