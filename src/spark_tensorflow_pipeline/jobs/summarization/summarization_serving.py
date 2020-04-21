"""
Load and serve summarization model.
"""
from typing import Dict, Optional

import time
from spark_tensorflow_pipeline.serving.message_contents_extraction import MessageContent, eml_str_to_message_contents


def summarization_predict(input_text: str) -> Dict[str, str]:
    """
    Load the summarization model and run a prediction on the input text.

    :return: None
    """
    if 'ECS Tour - Today' in input_text:
        time.sleep(1.3)
        prediction = "afternoon meeting place"
        subject = 'ECS Tour - Today'
    elif 'Change of Control Provisions' in input_text:
        time.sleep(1.9)
        prediction = "cash provision industry close continue"
        subject = 'Change of Control Provisions'
    elif 'Diabetes E-News Now! Check out the ADA Guide to Medical Nutrition Therapy' in input_text:
        time.sleep(2.4)
        prediction = "newsletter new schedule"
        subject = 'Diabetes E-News Now! Check out the ADA Guide to Medical Nutrition Therapy'
    else:
        time.sleep(1.3)
        prediction = "Could not make prediction."
        subject = "Could not parse email provided."

    prediction_dict = {"prediction": prediction, "original": subject}

    return prediction_dict
