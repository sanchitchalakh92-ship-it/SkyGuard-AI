.PHONY: setup train run test benchmark demo clean help

help:
	@echo "SkyGuard AI: Real-Time Anomaly Detection for AWS (SIH 26073)"
	@echo "Available commands:"
	@echo "  make setup      - Install Python dependencies"
	@echo "  make train      - Train temporal, multivariate, and fusion models"
	@echo "  make run        - Start FastAPI backend with real-time stream and dashboard"
	@echo "  make test       - Run unit tests and end-to-end integration tests"
	@echo "  make demo       - Execute 55C anomaly detection scenario"
	@echo "  make benchmark  - Run quantitative evaluation against baseline models"
	@echo "  make clean      - Clean temporary cache and log files"

setup:
	pip install -r requirements.txt

train:
	python -m src.models.train_all

run:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

demo:
	python examples/demo_55C_scenario.py

benchmark:
	python examples/run_benchmark.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
