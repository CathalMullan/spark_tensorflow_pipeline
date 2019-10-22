###
This should give a good overview of the problem to be solved.

Goals and high-level requirements should be stated clearly.

A description of a software / hardware development environment may be given.

Verifiable criteria against which the success of the project is to be judged should be identified (e.g. features of the
software and experimental results).

You may state these as acceptance tests, if preferred.

A Gantt chart or table which outlines the expected project development plan should be given (with major milestones and
deliverable's highlighted).

Appropriate references should be provided.

Include the URL for the gitLab repo. (max: 4 pages)

# Project Description and Work Plan

### Repository
https://gitlab.eeecs.qub.ac.uk/40180175/nlp_emails

### Problem Statement
Something along the lines of improving productivity etc.

...

### Solution Description
Creation of a pipeline to process bulk email files (.eml).

These emails will need to be scrubbed of any personally identifiable information (PII) before being processed.

This pipeline will consist of:
* Categorization of emails into topics such as 'Announcements', 'Business', 'E-Commerce', 'Social Media', etc.
* Identification of emails requiring a response, while identifying the degree of urgency required, such as high, normal
or low urgency.
* Summarizing of emails to an optimal length while maintaining relevance to matter at hand.

An API will serve the trained model, so the application can be offered in real-time at the email scanning stage.

A basic email client GUI will be created to showcase the change in workflow when managing emails with this mined
information.

### Overview of Core Technologies and Infrastructure
* Apache Spark: Distributed parallel processing framework
* Docker: Containerized builds.
* Kubernetes: Orchestrate the infrastructure.
* Kubernetes Operations (KOPS): Ease the creation of clusters.
* Amazon Web Services: Cloud computing facilitator of hardware.
* Amazon Simple Storage Service: Distributed data store.
* Terraform: Reproducible infrastructure as code.
* Helm: Provision instances with packages.
* Apache Airflow: Jobs scheduling and pipeline orchestration using directed acyclic graphs of execution.
* Jenkins: Enable continuous integration and continuous delivery.
* TensorFlow: Machine learning library with emphasis on Deep Learning.
* React: JavaScript UI library.

### Languages
* Python - PySpark
* TypeScript - React

### De-identification tool

#### Overview
Using non-domain specific pre-trained models, create a pipeline to consume raw email files, identify personally
identifiable information (PII) and replace this information with alternative 'fake' data.

The emails will likely be coming from AWS S3, but I'm not ruling out potential usage of the Hadoop File System (HDFS)
down the line. Apache Spark inherently uses Hadoop for file wrangling (specifically Apache Parquet files).

The data will be streamed directly into the pipeline using Spark Streaming and it's text file stream capabilities.
In terms of the scale of emails, my goal is to work at a scale equivalent to the daily amount of new emails ingested
by Proofpoint (number to be determined).

For entity recognition, I'll likely be using spaCy and one of it's pre-trained models, specifically the
en_core_web_lg model, which has been trained using the OntoNotes corpus.

Here's a basic example of spaCy identifying entities.

```
As you are aware, Enron utilizes temporary staffing services to satisfy
staffing requirements throughout the company.  For the past several months, a
project team, representing Enron's temporary staffing users, have researched
and evaluated alternative Managed Services programs to determine which source
would best meet our current and future needs in terms of quality, performance
and cost containment objectives.  The Business Unit Implementation Project
Team members are:

Laurie Koenig, Operations Management, EES
Carolyn Vigne, Administration, EE&CC
Linda Martin, Accounting & Accounts Payable, Corporate
Beverly Stephens, Administration, ENA
Norma Hasenjager, Human Resources, ET&S
Peggy McCurley, Administration, Networks
Jane Ellen Weaver, Enron Broadband Services
Paulette Obrecht, Legal, Corporate
George Weber, GSS

In addition, Eric Merten (EBS), Kathy Cook (EE&CC), Carolyn Gilley (ENA),
Larry Dallman (Corp/AP), and Diane Eckels (GSS) were active members of the
Selection Project Team.
```

```
As you are aware, ORG utilizes temporary staffing services to satisfy
staffing requirements throughout the company.  For DATE, a
project team, representing ORG,'s temporary staffing users, have researched
and evaluated alternative ORG programs to determine which source
would best meet our current and future needs in terms of quality, performance
and cost containment objectives.  The Business Unit Implementation Project
Team members are:

PERSON, ORG, ORG
PERSON, Administration, EE&CC
PERSON, ORG Payable, Corporate
PERSON, Administration, ENA
PERSON, ORG, ORG
PERSON, Administration, ORG
PERSON, ORG
PERSON, Legal, Corporate
PERSON, ORG

In addition, PERSON (ORG), PERSON (EE&CC), PERSON (ENA),
PERSON (ORG), and PERSON (ORG) were active members of ORG.
```

As for replacing the found entities, the Faker Python library should be suffice. If any domain specific/obscure text is
required to be generated, the Faker library can be extended using what it calls 'Providers', so perhaps a custom one
could be used.

Here is an example using Faker to replace the tags found by spaCy

```
As you are aware, Harrington-Perez utilizes temporary staffing services to satisfy
staffing requirements throughout the company.  For September, a
project team, representing Norris Ltd temporary staffing users, have researched
and evaluated alternative Pugh Group programs to determine which source
would best meet our current and future needs in terms of quality, performance
and cost containment objectives.  The Business Unit Implementation Project
Team members are:

Rita, Warren, Baker and Morgan, Lynch, Lee and Hickman
Patrick, Administration, EE&CC
Tamara, Stewart, Best and Mendez Payable, Corporate
Allison, Administration, ENA
Alison, Knox PLC, Sanders Inc
Barbara, Administration, Smith, Cruz and Burke
Vincent, Hobbs, Perez and Hull
Joseph, Legal, Corporate
Steven, Bishop-Moore

In addition, Victoria (Hughes Group), Cynthia (EE&CC), Erik (ENA),
Michael (King, Hawkins and Kelley), and Jennifer (Mcfarland, Palmer and Cervantes) were active members of Garcia, Davis and Norris.
```

By no means is this a perfect solution, as spaCy has clearly missed some 'personal' data.
But with the use of Faker, we can obscure the data with generated alternatives, making it far harder to identify an
individual with this email.
if the quality of the fakes were enriched, this effect would be amplified.

Once we are happy with these mails (and they've been approved by Proofpoints data controller), we can store these in a
separate S3 bucket. These are the emails which will be used to train the forthcoming models.

#### Acceptance Criteria
* Quality:
Approval by Proofpoints data controller on the legality side of things.

* Efficiency:
Minimum expectation is to be able to process a few million emails on a reasonably sized cluster (2-3 nodes) in less than an hour

#### References
N/A

### Categorization of emails

#### Overview
Standard classification of unstructured text into categories.
Since we won't know the categories ahead of time, this will encompass topic detection as well.
Overall a popular topic within NLP, specifically around medical text and social media.

* Likely to use the bag-of-words approach.
* Will use unsupervised learning.
* Dataset will simply be the email subject and body text.
* Clustering keywords will be important.
* Latent Dirichlet allocation (LDA) model seems the best fit.

The TensorFlow probability library looks useful for such work.

#### Acceptance Criteria
TBD

#### References
* [1] [Online Learning for Latent Dirichlet Allocation](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf)

### Identification of actionable emails

#### Overview
More domain specific challenge and likely to be the trickiest.
Work done on this topic for emails specifically, alongside social media text.
Once identified, sort by priority/urgency.
Perhaps may be more suitable to supervised learning.
In terms of the urgency detection, may end up being closer to sentiment analysis.
More angry/upset mail means more urgent? (Only half joking)

Reference [1] seems to approach this idea for social media with a focus on the found domains of the text, but this seems
rather closer to marketability research and a measure of brands success on media platforms.

Reference [2] is very much related to this projects goals. Further reading of this approach is necessary, but seems to
rely on splitting email text into what it calls email zones. Requires identification and understanding of speech
assertions and requests, and the syntactical difference between the two.

#### Acceptance Criteria
TBD

#### References
* [1] [Identifying Actionable Messages on Social Media](https://www.researchgate.net/publication/283532358_Identifying_Actionable_Messages_on_Social_Media)
* [2] [Detecting Emails Containing Requests for Action](https://www.aclweb.org/anthology/N10-1142.pdf)

### Summarizing of emails

#### Overview
Another more general goal of text modelling.
Unlikely to cause too much trouble.
The problem at hand can be viewed as unstructured text once again, and hence is fitting for unsupervised learning.
Will likely be phrase based, as opposed to word based for the categorization.

#### Acceptance Criteria
TBD

#### References
* [1] [A Survey of Unstructured Text Summarization Techniques](https://pdfs.semanticscholar.org/9064/f2a72907bdc78116ff07f551a0b2302ebcfc.pdf)

### Model Serving

#### Overview
TensorFlow has fairly extensive guides on serving their trained models.
Ideally model will be served as an API (likely created using Python), and hosted as part of Kubernetes cluster.
An alternative approach offered by TensorFlow is serving models through JavaScript, which may integrate well with the
email client component.

#### Acceptance Criteria
Likely being able to handle certain stress tests or benchmarking of requests.

#### References
* [1] [Serving Models](https://www.tensorflow.org/tfx/guide/serving)
* [2] [TensorFlow.js is a library for machine learning in JavaScript](https://www.tensorflow.org/js)
* [3] [TensorFlow On Spark](https://github.com/yahoo/TensorFlowOnSpark)

### Email Client

#### Overview
Creation of a progressive web app (PWA) / single page application (SPA) using React with TypeScript to view emails
while displaying the capabilities of the above models and showcasing the change in workflow (hopefully more efficient).

For example
* Automatically creating folders per topic.
* Allow searching per discovered topic.
* Have flag for urgency - can view actionable mails only
* Instead of showing a truncated version of the email body, show the summarized version when showcasing all mails.

Attempt to emulate GMail viewing capabilities, but worth noting this will only be used to view mail, will not be a fully
fledged email client.

#### Acceptance Criteria
TBD

#### References
N/A
...

### Gantt Chart
...
