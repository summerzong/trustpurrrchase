# Hugging Face Space (Docker SDK): builds the public GitHub repository.
# Put this file and README.md in a new Docker Space; add the secrets AIMO_API_KEY and GEMINI_API_KEY.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user
ARG REPO=https://github.com/summerzong/trustpurrrchase.git
ARG BRANCH=main
ARG CACHE_BUST=1
RUN git clone --depth 1 --branch "$BRANCH" "$REPO" app
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r demo/requirements.txt
ENV PATH=/home/user/.local/bin:$PATH HOST=0.0.0.0 PORT=7860 HOME_UI=chat PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "demo/start.sh"]
