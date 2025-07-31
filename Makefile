run:
	PYTHONPATH=$(pwd) python cli/run_forecast.py

forecast:
	PYTHONPATH=$(pwd) python cli/run_forecast.py --submit

ensemble:
	python cli/run_forecast.py --ensemble
