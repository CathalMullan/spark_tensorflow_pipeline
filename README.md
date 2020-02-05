# Spark TensorFlow Pipeline

## TODO
Workload expectations:
- **S**: 1 Day(s) - Small
- **M**: 2 Day(s) - Medium
- **L**: 4 Day(s) - Large

### In Progress
* M: Revisit initial eml pipeline step, try to integrate with Spark.
* S: Create Spark Structured Streaming equivalent of batch processor, awaiting events.
* L: Complete summarization task on email bodies.

### Short-term
* M: Read from Google Cloud in Minikube, with Spark job storing numpy array in bucket.
* L: Export an iterable model from TensorFlow training.
* S: Serve models over API.

### Medium-term
* L: Complete actionable email task on email bodies.
* M: Use Terraform to create cluster, run Horovod code.

### Long-term
* L: Complete CI Pipeline using master project.
* L: Create React PWA frontend which mimics email client.

### Optional
* S: Airflow scheduling of batch jobs.
* M: Export a TensorFlow.js friendly model.
* M: Serve model within JavaScript.
* L: Monitoring solutions through Grafana and Prometheus.
* M: Integrate `cathal.dev` domain.
* L: Continuous integration creating and destroying Terraform cluster.
* M: Serve models using WebAssembly.
* S: De-identification tool as a service.
* L: Hadoop integration for batch jobs.
* L: Consider using Vault and Consul for Kubernetes management.
* M: Unified CLI tool for downloading dependencies.
* L: Hybrid cloud streaming deployment over AWS and Google Cloud
* L: Perform audit on Kubernetes using kube-bench

### Title
Applying Natural Language Processing (NLP) techniques to detect topics, summarize, and identify actionable emails in a
distributed fashion using Spark and TensorFlow

### Supervisor
* Dr. Barry Devereux (b.devereux@qub.ac.uk)

### Project Setup
#### Create and Source Virtual Environment
* Run ````make venv````
* Run ````source venv/bin/activate````

#### Download and Extract Enron Dataset
* Run ````make enron````

#### Download spaCy Model
* Run ````make spacy````
