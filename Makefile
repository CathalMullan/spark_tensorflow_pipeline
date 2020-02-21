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

.PHONY: docker_image
docker_image:
	docker build -t spark_tensorflow_pipeline -f Dockerfile .

.PHONY: docker_publish
docker_publish:
	make docker_image

	docker tag spark_tensorflow_pipeline:latest gcr.io/distributed-email-pipeline/spark_tensorflow_pipeline:latest
	docker push gcr.io/distributed-email-pipeline/spark_tensorflow_pipeline:latest
