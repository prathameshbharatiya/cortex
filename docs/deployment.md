# Deployment

Cortex ships with a complete deployment stack: Docker, docker-compose, Kubernetes (Kustomize), and Helm.

## Docker

### Build

```bash
docker build -t cortex:latest .

# With specific extras
docker build --build-arg EXTRAS="grpc,redis,http,metrics" -t cortex:latest .
```

### Run gRPC server

```bash
docker run -p 50051:50051 \
  -e CORTEX_SERVER_PLATFORM_ID=arm_east_01 \
  -e CORTEX_API_KEYS=cx_live_your_key_here \
  -e CORTEX_REDIS_ENABLED=true \
  -e CORTEX_REDIS_URL=redis://redis:6379 \
  cortex:latest cortex-serve
```

### Run HTTP server

```bash
docker run -p 8765:8765 \
  -e CORTEX_SERVER_PLATFORM_ID=arm_east_01 \
  -e CORTEX_API_KEYS=cx_live_your_key_here \
  cortex:latest cortex-serve-http
```

---

## docker-compose (local / staging)

```bash
cp .env.example .env
# Edit .env: set CORTEX_API_KEYS, CORTEX_SERVER_PLATFORM_ID

docker-compose up -d

# Verify
cortex-health --server grpc://localhost:50051
cortex-health --server http://localhost:8765

# View logs
docker-compose logs -f cortex-grpc

# Stop
docker-compose down
```

The compose stack starts: `cortex-grpc`, `cortex-http`, Redis, Prometheus (`:9093`), Grafana (`:3000`, admin/admin).

---

## Kubernetes (Kustomize)

### Prerequisites

- `kubectl` configured for your cluster
- `kustomize` installed (`kubectl kustomize` or standalone)
- Container registry accessible from the cluster

### Deploy to staging

```bash
# Build and push image
docker build -t your-registry.io/cortex:latest .
docker push your-registry.io/cortex:latest

# Update the image reference
cd deploy/k8s/overlays/staging
kustomize edit set image cortex:latest=your-registry.io/cortex:latest

# Apply
kubectl apply -k deploy/k8s/overlays/staging

# Watch rollout
kubectl rollout status deployment/cortex-grpc -n cortex
```

### Deploy to production

```bash
docker build -t your-registry.io/cortex:0.1.0 .
docker push your-registry.io/cortex:0.1.0

cd deploy/k8s/overlays/production
kustomize edit set image cortex:latest=your-registry.io/cortex:0.1.0
kubectl apply -k deploy/k8s/overlays/production
kubectl rollout status deployment/cortex-grpc -n cortex --timeout=300s
```

### Configure secrets

```bash
# Generate an API key
cortex-keygen

# Create the secret
kubectl create secret generic cortex-secrets \
  --from-literal=CORTEX_API_KEYS=cx_live_your_key \
  -n cortex

# Or from a file
kubectl create secret generic cortex-secrets \
  --from-env-file=.env \
  -n cortex
```

### Verify

```bash
kubectl get pods -n cortex
kubectl logs -n cortex -l app=cortex-grpc --tail=50
kubectl exec -n cortex deploy/cortex-grpc -- cortex-health --json
```

---

## Helm

### Install

```bash
# From source
helm install cortex deploy/helm/cortex \
  --namespace cortex \
  --create-namespace \
  --set server.platformId=arm_east_01 \
  --set security.apiKeys=cx_live_your_key

# With custom values file
helm install cortex deploy/helm/cortex \
  --namespace cortex \
  --create-namespace \
  -f my-values.yaml
```

### Upgrade

```bash
helm upgrade cortex deploy/helm/cortex \
  --namespace cortex \
  --set image.tag=0.1.1
```

### Example production values file

```yaml
# production-values.yaml
image:
  repository: your-registry.io/cortex
  tag: "0.1.0"

replicaCount: 3

server:
  platformId:   arm_east_01
  deploymentId: prod
  logFormat:    json

gate:
  humanOverrideThreshold: 0.45
  replanThreshold:        0.55

sentinel:
  maxSpeedMs: 0.8
  maxForceN:  60.0

security:
  authRequired: true
  existingSecret: cortex-production-secrets   # managed by Vault

redis:
  enabled: true

metrics:
  enabled: true
  serviceMonitor:
    enabled: true   # requires Prometheus Operator

autoscaling:
  enabled:     true
  minReplicas: 3
  maxReplicas: 20
```

---

## Security in production

### API keys

Generate a separate key per robot:

```bash
cortex-keygen --robot-id arm_east_01 --perms certify+health
cortex-keygen --robot-id arm_west_02 --perms certify+health
cortex-keygen --robot-id operator_terminal --perms admin
```

Set them in your Kubernetes Secret:

```bash
kubectl create secret generic cortex-secrets \
  --from-literal=CORTEX_API_KEYS="cx_live_arm1...:arm_east_01:certify+health,cx_live_arm2...:arm_west_02:certify+health" \
  -n cortex
```

### mTLS (optional, highest security)

```bash
# Generate certificates
cortex-gen-certs \
  --output /etc/cortex/certs \
  --robots arm_east_01,arm_west_02

# Store server certificates in a K8s Secret
kubectl create secret generic cortex-tls-certs \
  --from-file=server.crt=/etc/cortex/certs/server/server.crt \
  --from-file=server.key=/etc/cortex/certs/server/server.key \
  --from-file=ca.crt=/etc/cortex/certs/ca/ca.crt \
  -n cortex

# Enable in your values file
security:
  tls:
    enabled: true
    mtls:    true
    existingCertSecret: cortex-tls-certs
```

---

## Observability

```bash
# Prometheus scrapes :9090/metrics on each pod
# Grafana: http://localhost:3000 (docker-compose) or set up ingress in K8s

# View real-time certification metrics
curl http://cortex-host:9090/metrics | grep cortex_certifications
```

Key metrics to monitor:

| Metric | Alert condition |
|---|---|
| `cortex_certifications_total{state="SAFE_HALT"}` | Any increase in production |
| `cortex_certification_latency_ms{quantile="0.99"}` | > 50ms |
| `cortex_memory_adapter_healthy{adapter="redis"}` | == 0 |
| `cortex_errors_total` | Any increase |

---

## Schema migrations

Before deploying a new Cortex version, always run:

```bash
# 1. Dry run first — no writes
cortex-migrate --adapter redis --url redis://prod-redis:6379 --dry-run

# 2. Run migration
cortex-migrate --adapter redis --url redis://prod-redis:6379

# 3. Verify
cortex-migrate --status
```

See [migrations.md](migrations.md) for the full migration guide.
