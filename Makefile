IMAGE_NAME = cloud-imperium-tts-demo-medium
CONTAINER_NAME = cloud-imperium-tts-demo-medium

.PHONY: build run

build:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile -t $(IMAGE_NAME) .

run:
	docker run --rm --name $(CONTAINER_NAME) -p 8081:80 -v ./samples:/app/samples/ $(IMAGE_NAME)