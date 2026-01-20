# Docker Deployment

Container deployment for production environments.

## Files

- **Dockerfile** - Container image definition
- **docker-compose.yml** - Multi-service orchestration

## Quick Start

### Single Container

```bash
# Build image
docker build -t anomaly-detection -f Dockerfile ..

# Run API
docker run -d -p 8000:8000 \
  -v $(pwd)/../core/anomaly_outputs:/app/anomaly_outputs \
  -v $(pwd)/../tests:/app/logs \
  --name anomaly-api \
  anomaly-detection
```

### Multi-Service (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services

The `docker-compose.yml` includes:

### 1. anomaly-api
REST API server on port 8000
- **Image**: Built from Dockerfile
- **Ports**: 8000:8000
- **Volumes**: models, logs
- **Health check**: Enabled

### 2. anomaly-mcp
MCP server for AI tools
- **Image**: Same as API
- **Mode**: stdio
- **Volumes**: models, logs

### 3. anomaly-batch
Scheduled batch processor
- **Image**: Same as API
- **Interval**: Configurable (default: 3600s)
- **Volumes**: models, logs, batch outputs

## Configuration

### Environment Variables

```yaml
environment:
  - MODEL_DIR=/app/anomaly_outputs
  - LOG_DIR=/app/logs
  - BATCH_INTERVAL=3600
  - PYTHONUNBUFFERED=1
```

### Volume Mounts

```yaml
volumes:
  - ../core/anomaly_outputs:/app/anomaly_outputs  # Models
  - ../tests:/app/logs                             # Input logs
  - ../batch/batch_outputs:/app/batch_outputs      # Results
```

## Health Checks

API includes health check:
```bash
# Check health
docker exec anomaly-api curl http://localhost:8000/health

# Docker health status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Production Deployment

### With HTTPS/TLS

Add nginx reverse proxy:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - anomaly-api
```

### Kubernetes

```bash
# Create deployment
kubectl apply -f k8s/deployment.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Scale replicas
kubectl scale deployment anomaly-detection --replicas=4
```

Example deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-detection
  template:
    metadata:
      labels:
        app: anomaly-detection
    spec:
      containers:
      - name: api
        image: anomaly-detection:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: models
          mountPath: /app/anomaly_outputs
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### Cloud Platforms

**AWS ECS:**
```bash
# Build and push
docker build -t your-repo/anomaly-detection .
docker push your-repo/anomaly-detection

# Deploy task definition
aws ecs create-service --cluster default --service-name anomaly-detection ...
```

**Google Cloud Run:**
```bash
# Deploy
gcloud run deploy anomaly-detection \
  --image gcr.io/your-project/anomaly-detection \
  --platform managed \
  --region us-central1
```

**Azure Container Instances:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name anomaly-detection \
  --image your-repo/anomaly-detection \
  --ports 8000
```

## Monitoring

### Logs

```bash
# View all service logs
docker-compose logs -f

# View specific service
docker-compose logs -f anomaly-api

# Last 100 lines
docker-compose logs --tail=100 anomaly-api
```

### Metrics

```bash
# Container stats
docker stats

# Specific container
docker stats anomaly-api
```

### Health Monitoring

```bash
# Check all services
docker-compose ps

# API health endpoint
curl http://localhost:8000/health
```

## Resource Limits

```yaml
services:
  anomaly-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Networking

### Custom Network

```yaml
networks:
  anomaly-net:
    driver: bridge

services:
  anomaly-api:
    networks:
      - anomaly-net
```

### External Access

```bash
# Access from host
curl http://localhost:8000/health

# Access from another container
curl http://anomaly-api:8000/health
```

## Data Persistence

### Named Volumes

```yaml
volumes:
  anomaly_outputs:
  logs:
  batch_outputs:

services:
  anomaly-api:
    volumes:
      - anomaly_outputs:/app/anomaly_outputs
      - logs:/app/logs
```

### Backup Volumes

```bash
# Backup models
docker run --rm -v anomaly-detection_anomaly_outputs:/data -v $(pwd):/backup \
  busybox tar czf /backup/models-backup.tar.gz /data

# Restore models
docker run --rm -v anomaly-detection_anomaly_outputs:/data -v $(pwd):/backup \
  busybox tar xzf /backup/models-backup.tar.gz -C /
```

## Troubleshooting

**Container won't start:**
```bash
# Check logs
docker logs anomaly-api

# Check events
docker events --filter container=anomaly-api
```

**Out of memory:**
```bash
# Increase memory limit
docker run -m 4g anomaly-detection
```

**Port already in use:**
```bash
# Use different port
docker run -p 8001:8000 anomaly-detection
```

## Documentation

- **Deployment Guide**: [`../docs/SCALING_GUIDE.md`](../docs/SCALING_GUIDE.md)
- **Production Setup**: [`../docs/README_SCALING.md`](../docs/README_SCALING.md)
