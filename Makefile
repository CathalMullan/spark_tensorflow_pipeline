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

.PHONY: test
test:
	pytest tests/ | tee logs/test.log

.PHONY: validate
validate:
	make lint | tee logs/validate.log
	make test | tee -a logs/validate.log
