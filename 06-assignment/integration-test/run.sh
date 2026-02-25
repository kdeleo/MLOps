
#!/usr/bin/env bash

export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

docker-compose up -d

sleep 10

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration || true

python integration_test.py

aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/

# docker-compose down

