# Distributed NLP Emails

### Title
Applying Natural Language Processing (NLP) techniques to detect topics, summarize, and identify actionable emails in a
distributed fashion using Spark and TensorFlow

### Supervisor(s)
* Machine Learning: Dr. Barry Devereux (b.devereux@qub.ac.uk)
* Infrastructure: Dr. David Cutting (d.cutting@qub.ac.uk)

### Summary
Creation of a pipeline to process bulk email files (.eml).

These emails will be provided by Proofpoint, an email security company with whom I've been working for over the last
year.
All emails are required to be scrubbed of any personally identifiable information (PII) before being processed, in
order to comply with regulations. All PII which is identified will either be removed or replaced with generated
alternatives.

This pipeline will consist of:
* Topic detection of emails into topics such as 'Announcements', 'Business', 'E-Commerce', 'Social Media', etc.
* Identification of emails requiring a response, while identifying the degree of urgency required, such as high, normal
or low urgency.
* Summarization of emails to an optimal length while maintaining relevance to matter at hand.

An API will serve the trained model, so the application can be offered in real-time at the email scanning stage.
The model may also be served directory in the browser.

A basic email client GUI will be created to showcase the change in workflow when managing emails with this mined
information.


### Project Setup
#### Create and Source Virtual Environment
* Run ````make venv````
* Run ````source venv/bin/activate````

#### Download and Extract Enron Dataset
* Run ````make enron````

#### Download spaCy Model
* Run ````make spacy````
