# TrustPurrrchase online demo: any Docker host (Render, Railway, Fly.io, Hugging Face Spaces, ...).
# Set AIMO_API_KEY (required) and GEMINI_API_KEY (optional) as secrets / environment variables.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt
COPY --chown=user . .
USER user
ENV HOST=0.0.0.0 PORT=7860 HOME_UI=chat PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "demo/start.sh"]
