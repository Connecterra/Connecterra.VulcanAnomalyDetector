FROM python:3.10-slim

# Install dependencies

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ssh \
        git

RUN mkdir -p -m 0600 $HOME/.ssh && ssh-keyscan github.com >> $HOME/.ssh/known_hosts

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=ssh,id=github_key pip install -r requirements.txt

COPY . .

CMD ["python", "src/evaluate.py"]