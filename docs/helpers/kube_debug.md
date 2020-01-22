# Kubernetes Debug
Standard procedure to follow when debugging Kubernetes problems.

## Prerequisites
* Connection to Kubernetes cluster

## Using Basic Image
Create a basic Ubuntu pod within the cluster.

```
kubectl run debug --image=ubuntu:xenial -- bash -c "sleep 1000000000;"
```

## Using Local Image
Create a pod using a local Docker image.

```
kubectl run debug --image=[...] --image-pull-policy Never -- bash -c "sleep 1000000000;"
```

## SSH into Pod
Connect to it

```
kubectl exec -it deployment/debug -- /bin/bash
```

If you're pod is an alpine

## Networking Debug
Install `telnet`, `dig`

```
apt-get -y update
apt-get -y install telnet dnsutils
```

Attempt to ping the ip of another resource

```
telnet ...
```

Or view the addresses of instances

```
dig [...].svc.cluster.local
```

## Tidy Up
Remember to delete your pod.

```
kubectl delete deployment/debug
```
