# Distributed NLP Emails

## TODO
Workload expectations:
- **S**: 1 Day(s) - Small
- **M**: 2 Day(s) - Medium
- **L**: 4 Day(s) - Large

### Short-term
* S: Create individual bucket specifically for testing data.
* S: Upload small Parquet files / email files to bucket, use for testing.
* M: Read from Google Cloud in Minikube, with Spark job storing numpy array in bucket.
* M: Get Horovod working with TensorFlow in Minikube.
* L: Export an iterable model from TensorFlow training.
* L: Validate iteration of model across a number of runs.

### Medium-term
* L: Complete summarization task on email bodies.
* M: Export a TensorFlow.js friendly model.
* M: Use Terraform to create cluster, run Horovod code.
* S: Create Spark Structured Streaming equivalent of batch processor, awaiting events.
* S: Create event producer to crawl web for public emails. (scrapy)

### Long-term
* L: Complete actionable email task on email bodies.
* L: Create React PWA frontend which mimics email client.
* S: Serve models over API.
* M: Serve model within JavaScript.
* S: Airflow scheduling of batch jobs.
* L: Monitoring solutions through Grafana and Prometheus.

### Optional
* M: Revisit initial eml pipeline step, try to integrate with Spark.
* M: Integrate `cathal.dev` domain.
* L: Continuous integration creating and destroying Terraform cluster.
* M: Serve models using WebAssembly.
* S: De-identification tool as a service.
* L: Hadoop integration for batch jobs.
* L: Consider using Vault and Consul for Kubernetes management.
* M: Unified CLI tool for downloading dependencies.
* L: Hybrid cloud streaming deployment over AWS and Google Cloud

### Outcome
- Min days of work: **40 Days** - **February 23, 2020**
- Max days of work: **63 Days** - **April 15, 2020**

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
