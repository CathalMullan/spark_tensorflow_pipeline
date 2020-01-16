.PHONY: venv
env:
	./scripts/init_environment.sh

.PHONY: hook
hook:
	pre-commit autoupdate
	pre-commit install

.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: test
test:
	pytest tests/

.PHONY: validate
validate:
	make lint
	make test

.PHONY: cluster-up
cluster-up:
	./scripts/create_cluster.sh

.PHONY: cluster-down
cluster-down:
	./scripts/delete_cluster.sh
