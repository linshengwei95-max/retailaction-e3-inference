# CPU inference image. Base: the Kaggle Python image digest used for the earlier private CPU runs (torch 2.11 CPU, torchvision 0.26, numpy; no GPU needed).
# The only addition is PyAV 16.0.1 (video decoding), pinned to the version used when the submitted predictions were produced. No other package is installed.
FROM gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a
ENV CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --no-deps av==16.0.1
WORKDIR /workspace
COPY infer.py model.py contract.py compare.py ./
COPY weights/ ./weights/
ENTRYPOINT ["python", "-B", "/workspace/infer.py"]
