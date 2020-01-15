#!/usr/bin/env bash
# Create test Kubernetes cluster.

# cd to project root
cd "$(dirname "${0}")" || exit
cd ../

# Start up Minikube with mounted Data directory and attach Kubectl
minikube start --memory 8192 --cpus 4

# Add Permissions to Default Service Account
kubectl create clusterrolebinding default --clusterrole=edit --serviceaccount=default:default --namespace=default

# Build Spark Docker Image
export SPARK_HOME=/opt/spark-2.4
(cd $SPARK_HOME && ./bin/docker-image-tool.sh -m -t spark build)

# Verify Image
eval $(minikube docker-env)
docker image ls

# Run Test Job
export CLUSTER_IP=k8s://https://$(minikube ip):8443
spark-submit \
    --master $CLUSTER_IP \
    --deploy-mode cluster \
    --name spark-pi \
    --class org.apache.spark.examples.SparkPi \
    --conf spark.executor.instances=2 \
    --conf spark.kubernetes.container.image=spark:spark \
    local:///opt/spark/examples/jars/spark-examples_2.11-2.4.4.jar

# Verify Answer
kubectl logs spark-pi-1548218924109-driver | grep "Pi is roughly"
