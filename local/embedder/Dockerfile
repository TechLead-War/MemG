FROM python:3.11-slim

WORKDIR /app

COPY local/embedder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/embedder.proto /proto/embedder.proto
RUN python -m grpc_tools.protoc \
    -I/proto \
    --python_out=/app \
    --grpc_python_out=/app \
    /proto/embedder.proto

COPY local/embedder/ .

EXPOSE 50051

ENTRYPOINT ["python", "server.py"]
CMD ["--model", "all-MiniLM-L6-v2"]
