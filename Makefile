.PHONY: venv
venv:
	./scripts/init_environment.sh

.PHONY: hook
hook:
	pre-commit autoupdate
	pre-commit install

.PHONY: lint
lint:
	pre-commit run --all-files | tee logs/lint.log
