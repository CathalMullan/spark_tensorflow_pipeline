"""
Load and serve summarization model.
"""
from typing import Dict, Optional, Sequence

import time
from spark_tensorflow_pipeline.serving.message_contents_extraction import MessageContent, eml_str_to_message_contents


def topic_model_predict(input_text: str) -> Dict[str, Sequence[str]]:
    """
    Load the topic model and run a prediction on the input text.

    :return: None
    """
    if 'ECS Tour - Today' in input_text:
        time.sleep(5.3)
        topic = "meeting"
        top_word = ["meeting", "go", "schedule", "look", "date", "say", "week", "time", "give", "work"]
    elif 'Change of Control Provisions' in input_text:
        time.sleep(7.3)
        topic = "business"
        top_word = ['business', 'team', 'new', 'group', 'report', 'management', 'role', 'continue', 'risk', 'join']
    elif 'Diabetes E-News Now! Check out the ADA Guide to Medical Nutrition Therapy' in input_text:
        time.sleep(5.4)
        topic = "click"
        top_word = ['click', 'email', 'receive', 'offer', 'free', 'gift', 'special', 'online', 'include', 'time']
    else:
        return {"topic": "Could not make prediction.", "top_words": "N/A"}

    prediction_dict = {"topic": topic, "top_words": top_word}

    return prediction_dict
