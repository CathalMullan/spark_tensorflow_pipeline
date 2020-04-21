"""
Serving API for predictions.
"""
from typing import Any

import uvicorn  # nosec
from fastapi import FastAPI, Request

from spark_tensorflow_pipeline.jobs.summarization.summarization_serving import summarization_predict
from spark_tensorflow_pipeline.jobs.topic_model.topic_model_serving import topic_model_predict

APP = FastAPI()


@APP.get("/summarization")
async def summarization(request: Request) -> Any:  # type: ignore
    """
    Summarization prediction.

    :param request: raw email file
    :return: prediction string
    """
    body_bytes: bytes = await request.body()
    return summarization_predict(body_bytes.decode(errors="replace"))


@APP.get("/topic")
async def topic(request: Request) -> Any:  # type: ignore
    """
    Summarization prediction.

    :param request: raw email file
    :return: prediction string
    """
    body_bytes: bytes = await request.body()
    return topic_model_predict(body_bytes.decode(errors="replace"))


if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=8000)  # nosec
