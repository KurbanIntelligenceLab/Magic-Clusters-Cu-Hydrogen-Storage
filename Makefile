lint:
	conda run -n aclWork flake8 .

format:
	conda run -n aclWork black .

isort:
	conda run -n aclWork isort .

run_ci: format isort lint