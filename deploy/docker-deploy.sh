#!/usr/bin/env bash
# Docker 部署入口：构建/启动 quillrag + Qdrant，并提供常用运维命令。
# 用法：
#   bash deploy/docker-deploy.sh up
#   bash deploy/docker-deploy.sh logs
#   bash deploy/docker-deploy.sh health

set -euo pipefail

ACTION="${1:-up}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "未找到 docker compose 或 docker-compose，请先安装 Docker Compose。"
    exit 1
  fi
}

ensure_env() {
  if [ ! -f .env ]; then
    cp .env.example .env
    echo "已从 .env.example 创建 .env，请填入 EMBEDDING_API_KEY 等必要配置后重新执行。"
    exit 1
  fi
}

wait_health() {
  local url="${QUILLRAG_HEALTH_URL:-http://127.0.0.1:8001/health}"
  local tries="${HEALTH_RETRIES:-30}"

  echo "等待服务健康：${url}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      echo "健康检查通过。"
      curl -s "$url"
      echo
      return 0
    fi
    echo "等待启动中... (${i}/${tries})"
    sleep 2
  done

  echo "健康检查未通过，请查看日志：bash deploy/docker-deploy.sh logs"
  return 1
}

case "$ACTION" in
  up)
    ensure_env
    compose up -d --build
    wait_health
    ;;
  start)
    ensure_env
    compose up -d
    wait_health
    ;;
  build)
    ensure_env
    compose build
    ;;
  restart)
    ensure_env
    compose restart quillrag
    wait_health
    ;;
  stop)
    compose stop
    ;;
  down)
    compose down
    ;;
  reset)
    compose down -v
    ;;
  ps|status)
    compose ps
    ;;
  logs)
    compose logs -f --tail="${2:-200}" quillrag
    ;;
  qdrant-logs)
    compose logs -f --tail="${2:-200}" qdrant
    ;;
  health)
    wait_health
    ;;
  *)
    echo "用法: bash deploy/docker-deploy.sh {up|start|build|restart|stop|down|reset|status|logs [N]|qdrant-logs [N]|health}"
    echo ""
    echo "  up           构建镜像并启动 quillrag + Qdrant"
    echo "  start        使用已有镜像启动"
    echo "  build        只构建 quillrag 镜像"
    echo "  restart      重启 quillrag 容器"
    echo "  stop         停止容器，保留数据"
    echo "  down         删除容器，保留 volume 数据"
    echo "  reset        删除容器和 volume 数据"
    echo "  status       查看容器状态"
    echo "  logs N       跟踪 quillrag 最近 N 行日志，默认 200"
    echo "  qdrant-logs  跟踪 Qdrant 日志"
    echo "  health       调用 /health"
    exit 1
    ;;
esac
