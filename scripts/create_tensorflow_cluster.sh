#!/usr/bin/env bash
# Create test Kubernetes cluster.

# cd to project root
cd "$(dirname "${0}")" || exit
cd ../

# Start up Minikube and attach Kubectl
minikube config set memory 4096
minikube config set cpus 2
minikube config set disk-size 60GB

minikube start
eval $(minikube docker-env)

# Create custom Service Account
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default

# Build base image (spark-py:spark)
export SPARK_HOME=/opt/spark-3.0
(cd ${SPARK_HOME} && ./bin/docker-image-tool.sh -t spark -p ./kubernetes/dockerfiles/spark/bindings/python/Dockerfile build)

# Build custom image (dnlp-pyspark)
docker build . -t dnlp-pyspark

# Test job with
spark-submit \
    --master k8s://https://$(minikube ip):8443 \
    --deploy-mode cluster \
    --name processing \
    --conf spark.executor.instances=2 \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=dnlp-pyspark \
    /src/distributed_nlp_emails/job/topic_modelling_processing.py

# View Spark Dashboard
kubectl get pod
kubectl port-forward <driver-pod-name> 4040:4040
open -n -a "Google Chrome" --args "--new-tab" http://localhost:4040
