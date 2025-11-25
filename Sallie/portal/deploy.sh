#!/bin/bash

# Safer bash settings
set -euo pipefail

# Flags added:
#   --domain your.domain.com   Enable HTTPS via Caddy + Let's Encrypt (requires public DNS & 80/443)
#   --email you@example.com    ACME contact email (recommended with --domain)
#   --local-https              Enable HTTPS locally using Caddy's internal CA (self-signed)
#   --no-build                 Skip calling build.sh (use previously-built images)
#   --image-name name          Specify podman image name (default: rules-portal)
#   --tag tag                 Specify podman image tag (default: latest)

# Parse arguments
PUSH=false
DOMAIN=""
EMAIL=""
LOCAL_TLS=false
NO_BUILD=false
IMAGE_NAME="rules-portal"
TAG="latest"
MODEL_HOME=""
MODEL_HOME_SET=false

# Track if image-name/tag were set by CLI
IMAGE_NAME_SET=false
TAG_SET=false
INTERNAL_DOMAIN_TLS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH=true
      shift
      ;;
    --domain)
      DOMAIN="$2"
      shift 2
      ;;
    --email)
      EMAIL="$2"
      shift 2
      ;;
    --local-https)
      # Use Caddy's internal CA for local HTTPS (self-signed)
      LOCAL_TLS=true
      shift
      ;;
    --no-build)
      NO_BUILD=true
      shift
      ;;
    --image-name)
      IMAGE_NAME="$2"
      IMAGE_NAME_SET=true
      shift 2
      ;;
    --tag)
      TAG="$2"
      TAG_SET=true
      shift 2
      ;;
    --home)
      MODEL_HOME="$2"
      MODEL_HOME_SET=true
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--push] [--domain your.domain.com --email you@example.com] [--local-https] [--no-build] [--image-name name] [--tag tag]"
      echo "  --push           Push images to podman Hub."
      echo "  --domain         Public domain to enable automatic HTTPS with Let's Encrypt via Caddy."
      echo "  --email          Contact email for ACME/Let's Encrypt registration."
      echo "  --local-https    Enable HTTPS with Caddy's internal CA (self-signed) for local/dev use."
      echo "  --no-build       Skip calling build.sh (use previously-built images)."
      echo "  --image-name     Specify podman image name (default: rules-portal)."
      echo "  --tag            Specify podman image tag (default: latest)."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; exit 1;
      ;;
  esac
done


# Determine model home default
if [[ "$MODEL_HOME_SET" != true ]]; then
  MODEL_HOME="$HOME/.model"
fi

# Neo4j data directory on host (alongside MODEL_HOME)
NEO4J_DATA_DIR="$(dirname "$MODEL_HOME")/neo4j-data"

# Ensure Neo4j data directory exists
mkdir -p "$NEO4J_DATA_DIR"

# PCPT config directory on host (used for ~/.pcpt inside container)
PCPT_HOST_HOME="${HOME}/.pcpt"
mkdir -p "${PCPT_HOST_HOME}"

# Ensure host PCPT log directory exists
mkdir -p "${PCPT_HOST_HOME}/log"

# PCPT in-container paths
PCPT_PROGRAM_PATH="/app/pcpt"
PCPT_CONFIG_PATH="/root/.pcpt/config"
PCPT_PROMPTS_PATH="/app/pcpt/prompts"
PCPT_HINTS_PATH="/app/pcpt/hints"
PCPT_LOG_PATH="/root/.pcpt/log"
PCPT_FILTERS_PATH="/app/pcpt/filters"

# AWS settings for PCPT – auto-detect profile/region from ~/.aws/config (same behavior as pcpt.sh)
AWS_CONFIG_FILE="$HOME/.aws/config"
PROFILE_FROM_CONFIG=""
REGION_FROM_CONFIG=""

if [[ -f "$AWS_CONFIG_FILE" ]]; then
  PROFILE_FROM_CONFIG=$(awk '/^\[profile / {gsub(/^\[profile /,""); gsub(/\]/,""); print; exit}' "$AWS_CONFIG_FILE")
  if [[ -z "$PROFILE_FROM_CONFIG" ]]; then
    PROFILE_FROM_CONFIG="default"
  fi
  REGION_FROM_CONFIG=$(awk -v profile="$PROFILE_FROM_CONFIG" '
    $0 ~ "\\[profile "profile"\\]" {found=1; next}
    /^\[profile / {found=0}
    found && $1 == "region" {print $3; exit}
  ' "$AWS_CONFIG_FILE")
fi

if [[ -z "$PROFILE_FROM_CONFIG" ]]; then
  PROFILE_FROM_CONFIG="default"
fi

if [[ -z "$REGION_FROM_CONFIG" ]]; then
  REGION_FROM_CONFIG="us-east-2"
fi

AWS_PROFILE_EFFECTIVE="$PROFILE_FROM_CONFIG"
AWS_REGION_EFFECTIVE="$REGION_FROM_CONFIG"

# Interactive prompt for IMAGE_NAME and TAG if not set by CLI and if running interactively (not in CI)
if [ -t 0 ]; then
  if [[ "$IMAGE_NAME_SET" != true ]]; then
    read -p "Enter image name [${IMAGE_NAME}]: " _in
    if [[ -n "$_in" ]]; then
      IMAGE_NAME="$_in"
    fi
  fi
  if [[ "$TAG_SET" != true ]]; then
    read -p "Enter image tag [${TAG}]: " _in
    if [[ -n "$_in" ]]; then
      TAG="$_in"
    fi
  fi
fi

# Resolve conflict between --domain and --local-https
if [[ -n "$DOMAIN" && "$LOCAL_TLS" == true ]]; then
  echo "⚠️  Both --domain and --local-https provided. Using internal TLS for domain: $DOMAIN"
  INTERNAL_DOMAIN_TLS=true
fi

# Generate build number (timestamp based)
BUILD_NUMBER=$(date +%y%m%d%H%M)

NETWORK_NAME="rules-portal-net"
PROXY_NAME="rules-portal-proxy"

# Neo4j configuration
NEO4J_NAME="rules-portal-neo4j"
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687

# Configuration
CONTAINER_NAME="rules-portal-app"
PORT=3201
# NOTE: The app healthcheck expects /healthz to return 200. Adjust the path in --health-cmd if your app uses a different endpoint.


echo "🚀 Starting deployment of Rules Portal..."

# Prepare PCPT artifacts under .tmp/pcpt-core for Docker build
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PCPT_SRC_DIR="${ROOT_DIR}/pcpt-core"
PCPT_TMP_DIR="${SCRIPT_DIR}/.tmp/pcpt-core"

echo "🧩 Preparing PCPT artifacts for image build..."
if [[ ! -d "${PCPT_SRC_DIR}" ]]; then
  echo "❌ PCPT source directory not found at ${PCPT_SRC_DIR}"
  echo "   Ensure pcpt-core is checked out alongside rules-portal."
  exit 1
fi

# Recreate .tmp/pcpt-core
rm -rf "${PCPT_TMP_DIR}"
mkdir -p "${PCPT_TMP_DIR}"

# Copy PCPT core files into .tmp so Dockerfile can COPY from .tmp/pcpt-core/...
cp "${PCPT_SRC_DIR}/requirements.txt" "${PCPT_TMP_DIR}/"
cp "${PCPT_SRC_DIR}/pcpt.spec"        "${PCPT_TMP_DIR}/"

cp -R "${PCPT_SRC_DIR}/code"    "${PCPT_TMP_DIR}/"
cp -R "${PCPT_SRC_DIR}/config"  "${PCPT_TMP_DIR}/"
cp -R "${PCPT_SRC_DIR}/hints"   "${PCPT_TMP_DIR}/"
cp -R "${PCPT_SRC_DIR}/prompts" "${PCPT_TMP_DIR}/"
cp -R "${PCPT_SRC_DIR}/shell"   "${PCPT_TMP_DIR}/"

echo "✅ PCPT artifacts staged in ${PCPT_TMP_DIR}"

# Optionally build images (default: build)
if [[ "$NO_BUILD" != true ]]; then
  echo "🧱 Running build.sh (use --no-build to skip)..."
  if [[ "$PUSH" == true ]]; then
    "$(dirname "$0")/build.sh" --push
  else
    "$(dirname "$0")/build.sh"
  fi
else
  echo "⏭️  Skipping build (per --no-build)"
fi

echo "🔗 Ensuring podman network $NETWORK_NAME exists..."
if ! podman network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
  podman network create "$NETWORK_NAME"
fi

# Stop and remove existing Neo4j container if it exists
echo "🛑 Stopping existing Neo4j container..."
podman stop $NEO4J_NAME 2>/dev/null || true

echo "🗑️  Removing existing Neo4j container..."
podman rm $NEO4J_NAME 2>/dev/null || true

echo "🧠 Starting Neo4j graph database on ports $NEO4J_HTTP_PORT (HTTP) and $NEO4J_BOLT_PORT (Bolt)..."
echo "📁 Neo4j data will be stored in: $NEO4J_DATA_DIR"

podman run -d \
  --name $NEO4J_NAME \
  --network $NETWORK_NAME \
  -p ${NEO4J_HTTP_PORT}:7474 \
  -p ${NEO4J_BOLT_PORT}:7687 \
  -v "${NEO4J_DATA_DIR}:/data" \
  -e NEO4J_AUTH=neo4j/rulesportal \
  neo4j:2025.10.1

if [ $? -ne 0 ]; then
  echo "❌ Failed to start Neo4j container!"
  exit 1
fi

# Stop and remove existing app container if it exists
echo "🛑 Stopping existing container..."
podman stop $CONTAINER_NAME 2>/dev/null || true

echo "🗑️  Removing existing container..."
podman rm $CONTAINER_NAME 2>/dev/null || true

# Decide run mode (direct HTTP vs behind Caddy HTTPS)
USE_PROXY=false
if [[ -n "$DOMAIN" || "$LOCAL_TLS" == true ]]; then
  USE_PROXY=true
fi

if [[ "$USE_PROXY" == true ]]; then
  echo "🚀 Starting app container on private network $NETWORK_NAME (no host port published)..."
  podman run -d \
      --name $CONTAINER_NAME \
      --network $NETWORK_NAME \
      -e NEO4J_URI="bolt://$NEO4J_NAME:$NEO4J_BOLT_PORT" \
      -v "${MODEL_HOME}:/app/public" \
      -v "${PCPT_HOST_HOME}:/root/.pcpt" \
      -v "$HOME/.aws:/root/.aws:ro" \
      -e AWS_PROFILE="$AWS_PROFILE_EFFECTIVE" \
      -e AWS_REGION="$AWS_REGION_EFFECTIVE" \
      -e AWS_SDK_LOAD_CONFIG=1 \
      -e AWS_EC2_METADATA_DISABLED=true \
      -e PCPT_PROGRAM_PATH="$PCPT_PROGRAM_PATH" \
      -e PCPT_CONFIG_PATH="$PCPT_CONFIG_PATH" \
      -e PCPT_PROMPTS_PATH="$PCPT_PROMPTS_PATH" \
      -e PCPT_HINTS_PATH="$PCPT_HINTS_PATH" \
      -e PCPT_LOG_PATH="$PCPT_LOG_PATH" \
      -e PCPT_FILTERS_PATH="$PCPT_FILTERS_PATH" \
      --health-cmd="wget -qO- http://127.0.0.1:$PORT/healthz || exit 1" \
      --health-interval=10s \
      --health-retries=3 \
      --health-timeout=3s \
      --read-only \
      --tmpfs /tmp:rw,nosuid,size=512m \
      --cap-drop ALL \
      --pids-limit=200 \
      --memory=512m \
      --cpus="1.0" \
      --restart unless-stopped \
      ${IMAGE_NAME}:${TAG}

  # TRACE: app container status
  if podman inspect $CONTAINER_NAME >/dev/null 2>&1; then
    podman inspect --format '🔍 TRACE: App container Health={{ (index .State "Health").Status }}' $CONTAINER_NAME 2>/dev/null || true
  fi
else
  echo "🚀 Starting app container on port $PORT (HTTP only)..."
  echo "(HTTP on host port 443; use http://localhost)"
  podman run -d \
      --name $CONTAINER_NAME \
      --network $NETWORK_NAME \
      -e NEO4J_URI="bolt://$NEO4J_NAME:$NEO4J_BOLT_PORT" \
      -p 443:$PORT \
      -v "${MODEL_HOME}:/app/public" \
      -v "${PCPT_HOST_HOME}:/root/.pcpt" \
      -v "$HOME/.aws:/root/.aws:ro" \
      -e AWS_PROFILE="$AWS_PROFILE_EFFECTIVE" \
      -e AWS_REGION="$AWS_REGION_EFFECTIVE" \
      -e AWS_SDK_LOAD_CONFIG=1 \
      -e AWS_EC2_METADATA_DISABLED=true \
      -e PCPT_PROGRAM_PATH="$PCPT_PROGRAM_PATH" \
      -e PCPT_CONFIG_PATH="$PCPT_CONFIG_PATH" \
      -e PCPT_PROMPTS_PATH="$PCPT_PROMPTS_PATH" \
      -e PCPT_HINTS_PATH="$PCPT_HINTS_PATH" \
      -e PCPT_LOG_PATH="$PCPT_LOG_PATH" \
      -e PCPT_FILTERS_PATH="$PCPT_FILTERS_PATH" \
      --health-cmd="wget -qO- http://127.0.0.1:$PORT/healthz || exit 1" \
      --health-interval=10s \
      --health-retries=3 \
      --health-timeout=3s \
      --read-only \
      --tmpfs /tmp:rw,nosuid,size=512m \
      --cap-drop ALL \
      --pids-limit=200 \
      --memory=512m \
      --cpus="1.0" \
      --restart unless-stopped \
      ${IMAGE_NAME}:${TAG}

  # TRACE: app container status
  if podman inspect $CONTAINER_NAME >/dev/null 2>&1; then
    podman inspect --format '🔍 TRACE: App container Health={{ (index .State "Health").Status }}' $CONTAINER_NAME 2>/dev/null || true
  fi
fi

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

# If HTTPS requested, (re)start Caddy reverse proxy
if [[ "$USE_PROXY" == true ]]; then
  echo "🧰 Preparing Caddy reverse proxy for HTTPS..."

  # Stop/remove any existing proxy
  podman stop $PROXY_NAME 2>/dev/null || true
  podman rm $PROXY_NAME 2>/dev/null || true

  # Write a temporary Caddyfile
  CADDYFILE_PATH=$(pwd)/Caddyfile.tmp
  echo "📄 Writing Caddyfile to $CADDYFILE_PATH"
  {
  if [[ -n "$EMAIL" && "$LOCAL_TLS" != true && "$INTERNAL_DOMAIN_TLS" != true ]]; then
    echo "  email $EMAIL"
  fi
  echo "  admin off"
  }
  if [[ "$INTERNAL_DOMAIN_TLS" == true ]]; then
  # Internal TLS for provided domain (self-signed for the domain)
  echo "$DOMAIN {"
  echo "  encode zstd gzip"
  echo "  header {"
  echo "    X-Content-Type-Options \"nosniff\""
  echo "    X-Frame-Options \"DENY\""
  echo "    Referrer-Policy \"no-referrer\""
  echo "    Content-Security-Policy \"default-src 'self'\""
  echo "  }"
  echo "  tls internal"
  echo "  log"
  echo "  reverse_proxy ${CONTAINER_NAME}:$PORT"
  echo "}"
elif [[ "$LOCAL_TLS" == true ]]; then
  # Local/dev HTTPS with Caddy internal CA on https://localhost
  echo "localhost {"
  echo "  encode zstd gzip"
  echo "  header {"
  echo "    X-Content-Type-Options \"nosniff\""
  echo "    X-Frame-Options \"DENY\""
  echo "    Referrer-Policy \"no-referrer\""
  echo "    Content-Security-Policy \"default-src 'self'\""
  echo "  }"
  echo "  tls internal"
  echo "  log"
  echo "  reverse_proxy ${CONTAINER_NAME}:$PORT"
  echo "}"
else
  # Public domain with Let's Encrypt
  echo "$DOMAIN {"
  echo "  encode zstd gzip"
  echo "  header {"
  echo "    Strict-Transport-Security \"max-age=31536000; includeSubDomains; preload\""
  echo "    X-Content-Type-Options \"nosniff\""
  echo "    X-Frame-Options \"DENY\""
  echo "    Referrer-Policy \"no-referrer\""
  echo "    Content-Security-Policy \"default-src 'self'\""
  echo "  }"
  echo "  tls {"
  echo "    protocols tls1.2 tls1.3"
  echo "  }"
  echo "  log"
  echo "  reverse_proxy ${CONTAINER_NAME}:$PORT"
  echo "}"
fi > "$CADDYFILE_PATH"

echo "🔍 TRACE: Final Caddyfile content =============================="
cat "$CADDYFILE_PATH"
echo "================================================================"

  echo "🚀 Starting Caddy proxy on ports 80/443..."
  podman run -d \
    --name $PROXY_NAME \
    --network $NETWORK_NAME \
    -p 80:80 \
    -p 443:443 \
    -v "$CADDYFILE_PATH":/etc/caddy/Caddyfile:ro \
    -v caddy_data:/data \
    -v caddy_config:/config \
    --health-cmd="wget -qO- http://127.0.0.1:80/ || exit 1" \
    --health-interval=10s \
    --health-retries=3 \
    --health-timeout=3s \
    --restart unless-stopped \
    caddy:2

# TRACE: show caddy container network bindings
if podman inspect $PROXY_NAME >/dev/null 2>&1; then
  podman inspect $PROXY_NAME --format '🔍 TRACE: Caddy container IP={{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' || true
  podman port $PROXY_NAME | sed 's/^/🔍 TRACE: /' || true
fi

  if [ $? -ne 0 ]; then
    echo "❌ Failed to start Caddy proxy!"
    echo "🔍 TRACE: Caddy logs (last 100 lines):"
    podman logs --tail 100 $PROXY_NAME 2>/dev/null || true
    exit 1
  fi

  if [[ "$INTERNAL_DOMAIN_TLS" == true ]]; then
    echo "✅ Internal HTTPS enabled via Caddy (self-signed) for domain: https://$DOMAIN"
    echo "   Note: Your browser will warn unless you trust Caddy's local CA."
  elif [[ "$LOCAL_TLS" == true ]]; then
    echo "✅ Local HTTPS enabled via Caddy (internal CA). Access: https://localhost"
    echo "   Note: Your browser will show a warning unless you trust Caddy's local CA."
  else
    echo "✅ Public HTTPS enabled via Caddy + Let's Encrypt for domain: https://$DOMAIN"
    echo "   Ensure DNS A/AAAA records point to this host and ports 80/443 are open."
  fi

# Optional: verify the certificate Caddy is serving (host-side, non-blocking)
SNI_HOST="$DOMAIN"
if [[ "$LOCAL_TLS" == true && "$INTERNAL_DOMAIN_TLS" != true ]]; then
  SNI_HOST="localhost"
fi

echo "🔍 TRACE: Probing HTTPS endpoint via host curl (SNI=$SNI_HOST)..."
# We hit 127.0.0.1:443 and set the Host header for SNI; max 3s per try, up to 5 tries
for i in 1 2 3 4 5; do
  if curl -skI --max-time 3 https://127.0.0.1/ -H "Host: $SNI_HOST" >/dev/null 2>&1; then
    echo "🔍 TRACE: curl reached Caddy on attempt #$i"
    break
  fi
  sleep 1
  if [[ $i -eq 5 ]]; then
    echo "⚠️  TRACE: curl couldn't reach https://$SNI_HOST on 127.0.0.1:443 after 5 tries (continuing)."
  fi
done

# If openssl is available on the host, print concise cert info with a 5s timeout if available
if command -v openssl >/dev/null 2>&1; then
  echo "🔍 TRACE: TLS cert (subject/issuer/dates) from host perspective:"
  if command -v timeout >/dev/null 2>&1; then
    timeout 5s bash -c "echo | openssl s_client -servername '$SNI_HOST' -connect 127.0.0.1:443 2>/dev/null | openssl x509 -noout -subject -issuer -dates" || true
  else
    bash -c "echo | openssl s_client -servername '$SNI_HOST' -connect 127.0.0.1:443 2>/dev/null | openssl x509 -noout -subject -issuer -dates" || true
  fi
else
  echo "ℹ️  TRACE: openssl not found on host; skipping cert details."
fi

else
  echo "🌐 Application is running over HTTP on port 443. Open: http://localhost (or http://localhost:443)"
  echo "🔒 For HTTPS on localhost, rerun: $0 --local-https"
fi

# Show container status
echo "📊 Container status:"
podman container ls --filter "name=$CONTAINER_NAME" --filter "name=$PROXY_NAME" --filter "name=$NEO4J_NAME"

echo "🧠 Neo4j is available on http://localhost:${NEO4J_HTTP_PORT} (browser) and bolt://localhost:${NEO4J_BOLT_PORT} (Bolt)."
echo "   Default credentials (dev only): neo4j / rulesportal"

echo "💡 TRACE tip: to follow logs, run: podman logs -f $PROXY_NAME (proxy) or podman logs -f $CONTAINER_NAME (app)"