.PHONY: classification

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	pre-commit run

docker-build: ./requirements/requirements-docker.txt
	docker build -t catalyst-classification:latest . -f docker/Dockerfile

docker-clean:
	rm -rf build/
	docker rmi -f catalyst-classification:latest
