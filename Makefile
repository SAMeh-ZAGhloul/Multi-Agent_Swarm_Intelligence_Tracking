# AEGIS-AI Makefile
# Multi-Agent Counter-UAS Swarm Tracking Platform

.PHONY: all install dev sim test lint clean help

# Default target
all: install lint test

# Install dependencies
install:
	@echo "Installing AEGIS-AI dependencies..."
	pip install -e ".[dev]"
	pre-commit install
	@echo "Installation complete!"

# Development mode - start all services
dev:
	@echo "Starting AEGIS-AI development environment..."
	@echo "Starting ingest service..."
	python -m services.ingest.main &
	@echo "Starting tracker service..."
	python -m services.tracker.main &
	@echo "Starting swarm intelligence service..."
	python -m services.swarm_intel.main &
	@echo "Starting coordinator service..."
	python -m services.coordinator.main &
	@echo "Starting gateway service..."
	python -m services.gateway.main &
	@echo "Starting Streamlit UI..."
	streamlit run ui/app.py
	@echo "All services started. Press Ctrl+C to stop."

# Run simulator
sim:
	@echo "Running AEGIS-AI swarm simulator..."
	python -c "from core.simulation import ScenarioRunner, ScenarioType; runner = ScenarioRunner(); runner.run_scenario(ScenarioType.SATURATION_ATTACK); print('Saturation Attack scenario complete!'); print('Total simulation time: 90 seconds'); print('Simulated drones per step: 50')"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=core --cov=services --cov-report=term-missing

# Run tests with benchmarks
test-bench:
	@echo "Running benchmark tests..."
	pytest tests/ -v -m benchmark --benchmark-only

# Run linting
lint:
	@echo "Running linters..."
	ruff check core services ui
	ruff format core services ui --check
	mypy core services

# Fix linting issues
format:
	@echo "Formatting code..."
	ruff check core services ui --fix
	ruff format core services ui

# Run type checking only
typecheck:
	@echo "Running MyPy..."
	mypy core services

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.prof" -delete
	rm -rf build dist htmlcov .coverage
	@echo "Clean complete!"

# Run single scenario test
scenario:
	@echo "Running scenario: $(SCENARIO)"
	python -c "\
from core.simulation import ScenarioRunner, ScenarioType; \
runner = ScenarioRunner(); \
runner.run_scenario(ScenarioType.$(SCENARIO)); \
print('Scenario:', runner.get_scenario_description(runner.current_scenario)); \
states = runner.simulator.step(0.1); \
for sid, s in states.items(): \
    print(f'  {sid}: {len(s.drones)} drones, behavior={s.behavior.name}, threat={s.threat_score:.2f}')"

# Start Streamlit UI only
ui:
	@echo "Starting Streamlit UI..."
	streamlit run ui/app.py

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

# Help
help:
	@echo "AEGIS-AI Makefile Commands"
	@echo ""
	@echo "  install     - Install all dependencies"
	@echo "  dev         - Start all services + Streamlit UI"
	@echo "  sim         - Run swarm simulator demo"
	@echo "  test        - Run test suite"
	@echo "  test-bench  - Run benchmark tests"
	@echo "  lint        - Run linters (ruff + mypy)"
	@echo "  format      - Auto-format code"
	@echo "  typecheck   - Run MyPy only"
	@echo "  clean       - Remove build artifacts"
	@echo "  scenario    - Run specific scenario (use SCENARIO=...)"
	@echo "  ui          - Start Streamlit UI only"
	@echo "  docker-*    - Docker build/up/down/logs"
	@echo ""
	@echo "Example:"
	@echo "  make scenario SCENARIO=SATURATION_ATTACK"
