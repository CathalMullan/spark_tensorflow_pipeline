.PHONY: hook
hook:
	pre-commit autoupdate
	pre-commit install

.PHONY: lint
lint:
	pre-commit run --all-files | tee logs/lint.txt

.PHONY: test
test:
	pytest tests/ | tee logs/test.txt

.PHONY: validate
validate:
	make lint
	make test

.PHONY: docker_image
docker_image:
	docker build -t spark_tensorflow_pipeline -f gpu.Dockerfile .

.PHONY: docker_publish
docker_publish:
	make docker_image
	docker tag spark_tensorflow_pipeline:latest gcr.io/distributed-email-pipeline/spark_tensorflow_pipeline:latest
	docker push gcr.io/distributed-email-pipeline/spark_tensorflow_pipeline:latest

.PHONY: dev_image
dev_docker_image:
	docker build -t dev_pipeline -f cpu.Dockerfile .

.PHONY: topic_dev
topic_dev:
	make dev_docker_image
	docker run \
		--env-file .env \
		-v ~/.config/gcloud/gcp_service_account.json:/etc/secrets/gcp_service_account.json \
		-it dev_pipeline \
			mpirun \
			--allow-run-as-root \
            -np \
            "2" \
            -bind-to \
            none \
            -map-by \
            slot \
            -x \
            NCCL_DEBUG=INFO \
            -x \
            LD_LIBRARY_PATH \
            -x \
            PATH \
            -mca \
            pml \
            ob1 \
            -mca \
            btl \
            ^openib \
            python \
            /app/src/spark_tensorflow_pipeline/jobs/topic_model/topic_model.py

.PHONY: summarization_dev
summarization_dev:
	make dev_docker_image
	docker run \
		--env-file .env \
		-v ~/.config/gcloud/gcp_service_account.json:/etc/secrets/gcp_service_account.json \
		-it dev_pipeline \
			mpirun \
			--allow-run-as-root \
            -np \
            "2" \
            -bind-to \
            none \
            -map-by \
            slot \
            -x \
            NCCL_DEBUG=INFO \
            -x \
            LD_LIBRARY_PATH \
            -x \
            PATH \
            -mca \
            pml \
            ob1 \
            -mca \
            btl \
            ^openib \
            python \
            /app/src/spark_tensorflow_pipeline/jobs/summarization/summarization_model.py
