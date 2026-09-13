#!/usr/bin/env bash
set -e

curl -sS -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features":["feature_1","feature_2","feature_3"]}'
