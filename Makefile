.PHONY: venv install run-api mlflow docker-build

venv:
.gitpython -m venv .venv
.git@echo "Activate with: .venv\Scripts\activate"

install:
.gitpip install -r requirements.txt

run-api:
.gituvicorn app.main:app --reload

mlflow:
.gitmlflow ui

docker-build:
.gitdocker build -t qosbuddy-5g .
