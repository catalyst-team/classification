.PHONY: classification

docker-build: ./requirements/requirements-docker.txt
	docker build -t catalyst-classification:latest . -f docker/Dockerfile

docker-clean:
	rm -rf build/
	docker rmi -f catalyst-classification:latest
