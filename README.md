# NLP Emails

### Title
Applying Natural Language Processing (NLP) techniques to categorize and summarize emails

### Supervisor(s)
* Machine Learning: Dr. Barry Devereux (b.devereux@qub.ac.uk)
* Infrastructure: Dr. David Cutting (d.cutting@qub.ac.uk)

### Summary
Creation of a pipeline to process bulk real-world email files (.eml) provided by Proofpoint.

These emails will need to be scrubbed of any personally identifiable information (PII) before being processed.

This pipeline will consist of:
* Categorization of emails into topics such as 'Announcements', 'Business', 'E-Commerce', 'Social Media', etc.
* Identification of emails requiring a response, while identifying the degree of urgency required, such as high, normal
or low urgency.
* Summarizing of emails to an optimal length while maintaining relevance to matter at hand.

An API will serve the trained model, so the application can be offered in real-time at the email scanning stage.

A basic email client GUI will be created to showcase the change in workflow when managing emails with this mined
information.

### Keywords
text mining, topic detection, classification, categorization
