# Metac Agent Agent - Tournament Optimization System

A sophisticated forecasting system designed for tournament-style prediction competitions, featuring advanced multi-agent ensemble methods, comprehensive research pipelines, and intelligent strategy optimization.

## 🚀 Features

### Core Capabilities
- **Tournament Orchestration**: Complete workflow from question ingestion through research, reasoning, prediction, ensemble aggregation, and submission
- **Multi-Agent Ensemble**: Sophisticated ensemble methods with Chain-of-Thought, Tree-of-Thought, and ReAct agents
- **Advanced Research Pipeline**: Multi-provider evidence gathering with credibility analysis and source validation
- **Strategy Optimization**: Tournament-specific strategy selection and timing optimization
- **Risk Management**: Comprehensive calibration and risk adjustment mechanisms

### System Architecture
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **Resilience**: Circuit breakers, retry strategies, and graceful degradation
- **Monitoring**: Comprehensive health checks, metrics, and distributed tracing
- **Scalability**: Async processing, caching, and concurrent question handling

### Interfaces
- **CLI**: Full command-line interface for question processing and tournament analysis
- **REST API**: Complete REST API with endpoints for processing, monitoring, and export
- **Backward Compatibility**: Seamless integration with existing main entry points

## 📋 Requirements

- Python 3.9+
- Poetry for dependency management
- Redis (optional, for caching)
- PostgreSQL (optional, for persistence)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/metac-agent-agent.git
cd metac-agent-agent
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Run tests to verify installation:
```bash
poetry run pytest
```

## 🚀 Quick Start

### Basic Usage

Process a single question:
```bash
python main.py --mode test_questions --use-optimization
```

Run in legacy mode:
```bash
python main.py --mode tournament --legacy-mode
```

### CLI Interface

Process a question with tournament optimization:
```bash
poetry run python -m src.presentation.cli process-question 123 --tournament-id 456
```

Analyze tournament strategy:
```bash
poetry run python -m src.presentation.cli analyze-strategy 456
```

Check system status:
```bash
poetry run python -m src.presentation.cli status
```

### REST API

Start the API server:
```bash
poetry run python -m src.presentation.rest_api
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### Agent Interface

Use the agent interface:
```bash
python main_agent.py --question question.json --use-optimization --tournament-id 456
```

## 📖 Documentation

### Architecture Overview

The system follows a clean architecture pattern with the following layers:

- **Domain Layer**: Core business logic, entities, and value objects
- **Application Layer**: Use cases and application services
- **Infrastructure Layer**: External integrations, caching, monitoring
- **Presentation Layer**: CLI, REST API, and legacy interfaces

### Key Components

1. **ProcessTournamentQuestion Use Case**: Main orchestrator for question processing
2. **ForecastingPipeline**: Coordinates research, reasoning, and prediction phases
3. **TournamentService**: Strategy analysis and optimization
4. **IntegrationService**: Backward compatibility with existing systems

### Configuration

The system supports multiple configuration methods:

- Environment variables
- Configuration files (`configs/`)
- Feature flags (`configs/feature_flags.json`)
- Command-line arguments

## 🧪 Testing

Run the full test suite:
```bash
poetry run pytest
```

Run specific test categories:
```bash
# Unit tests
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# End-to-end tests
poetry run pytest tests/e2e/

# Performance tests
poetry run pytest tests/performance/
```

## 📊 Monitoring

### Health Checks

The system includes comprehensive health checks:
- Database connectivity
- External API availability
- Cache system status
- Agent system health

### Metrics

Key metrics are collected and can be exported to monitoring systems:
- Processing times
- Success rates
- Cache hit rates
- Consensus strength
- Agent performance

### Logging

Structured logging with correlation IDs for distributed tracing:
- Request correlation across all components
- Performance metrics
- Error tracking and recovery

## 🔧 Development

### Project Structure

```
metac-agent-agent/
├── src/
│   ├── domain/              # Core business logic
│   ├── application/         # Use cases and services
│   ├── infrastructure/      # External integrations
│   └── presentation/        # Interfaces (CLI, API)
├── tests/                   # Test suites
├── configs/                 # Configuration files
├── scripts/                 # Deployment and utility scripts
└── docs/                    # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `poetry run pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Quality

The project maintains high code quality standards:
- Type hints throughout
- Comprehensive test coverage
- Linting with flake8 and black
- Security scanning
- Performance testing

## 🚀 Deployment

### Docker

Build and run with Docker:
```bash
docker build -t metac-agent-agent .
docker run -p 8000:8000 metac-agent-agent
```

### Kubernetes

Deploy to Kubernetes:
```bash
kubectl apply -f k8s/deployment-template.yaml
```

### Infrastructure as Code

Terraform configurations are provided in `infrastructure/terraform/`:
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

## 📈 Performance

The system is designed for high performance:
- Async processing throughout
- Intelligent caching strategies
- Connection pooling
- Concurrent question processing
- Memory optimization

Typical performance metrics:
- Question processing: 2-10 seconds
- Tournament analysis: 30-60 seconds
- API response time: <200ms
- Cache hit rate: >80%

## 🔒 Security

Security features include:
- Input validation and sanitization
- Rate limiting
- Credential management
- Audit logging
- Security middleware
- Vulnerability scanning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the test examples in `tests/`

## 🎯 Roadmap

- [ ] Advanced ML-based question categorization
- [ ] Real-time tournament leaderboard integration
- [ ] Enhanced visualization dashboard
- [ ] Mobile API endpoints
- [ ] Advanced ensemble methods
- [ ] Automated strategy backtesting

## 🏆 Acknowledgments

Built for tournament-style forecasting competitions with a focus on:
- Metaculus tournaments
- Prediction markets
- Forecasting competitions
- Research applications

---

**Note**: This system provides both advanced tournament optimization features and backward compatibility with existing forecasting workflows. Choose the mode that best fits your needs.
