#!/usr/bin/env bash
# Assemble a minimal Lambda source dir (package + requirements only), then build and deploy:
#   ./scripts/package.sh && sam build -t infra/template.yaml && sam deploy --guided
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/lambda && mkdir -p build/lambda
cp -R translator requirements.txt build/lambda/
find build/lambda -name '__pycache__' -prune -exec rm -rf {} +
echo "Lambda source ready in build/lambda"
