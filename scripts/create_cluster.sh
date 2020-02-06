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

# Build custom image
docker build . -t spark_tensorflow_pipeline

# Run job
...

# View Spark Dashboard
kubectl get pod
kubectl port-forward <driver-pod-name> 4040:4040
open -n -a "Google Chrome" --args "--new-tab" http://localhost:4040
