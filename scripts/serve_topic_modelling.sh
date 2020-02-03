#!/usr/bin/env bash
# Serve the trained Topic Modelling model.

# cd to project root
cd "$(dirname "${0}")" || exit
cd ../

# serve using Dockerfile
docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/models/topic_model:/models/topic_model" \
    -e MODEL_NAME=topic_model \
    tensorflow/serving

curl http://localhost:8501/v1/models/topic_model


curl -X POST http://localhost:8501/v1/models/topic_model:predict -d '{"processed_text": "hello"}'
