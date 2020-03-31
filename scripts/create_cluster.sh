#!/usr/bin/env bash
# Create test Kubernetes cluster.

# cd to project root
cd "$(dirname "${0}")" || exit
cd ../

# Start up Minikube and attach Kubectl
minikube config set memory 4096
minikube config set cpus 6
minikube config set disk-size 60GB
minikube config set vm-driver virtualbox
minikube config set kubernetes-version 1.17.0

minikube start
eval $(minikube docker-env)

# Build custom image
docker build . -t spark_tensorflow_pipeline

# Setup MPI
kubectl apply -f kubernetes/mpi_operator.yaml
kubectl get crd
kubectl create -f kubernetes/mpi_test_job.yaml
kubectl get mpijobs tensorflow-benchmarks
