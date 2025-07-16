lint:
	flake8 .

format:
	black .

isort:
	isort .

run_ci: format isort lint