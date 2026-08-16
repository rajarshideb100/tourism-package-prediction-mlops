FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user . /home/user/app

USER user

RUN pip install --upgrade pip && \
    pip install -r tourism_project/deployment/requirements.txt

EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "tourism_project/deployment/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--server.enableXsrfProtection=false"]
