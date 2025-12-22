INTERVAL=5  # 초 단위, 원하면 5, 30 등으로 조정

while true; do
  echo "==================== $(date '+%Y-%m-%d %H:%M:%S') ===================="
  echo "[Docker 컨테이너 리소스 사용량]"
  docker stats --no-stream

  echo
  echo "[GPU 사용량]"
  nvidia-smi

  echo
  sleep "$INTERVAL"
done
