# Deployment

## Docker Compose

From the project root:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

The dashboard is on `http://localhost:8501` and the API is on `http://localhost:8000/docs`.

The training container performs the full data/training pipeline. For a first build, it can take substantial CPU/RAM because TensorFlow trains both an LSTM and Transformer and GARCH is also fitted.

## Kubernetes

Build and push the image, replace `your-registry/volatility-inference:latest` in the manifests, then apply:

```bash
kubectl apply -f deployment/k8s/inference-deployment.yaml
kubectl apply -f deployment/k8s/dashboard-deployment.yaml
kubectl apply -f deployment/k8s/ingestion-cronjob.yaml
```

Training should be run as a controlled batch job or CI/CD step so that the resulting model artifacts are mounted/published before the inference deployment becomes ready.
