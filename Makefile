.PHONY: classification

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	bash ./bin/_check_codestyle.sh

docker-build: ./requirements/requirements_docker.txt
	docker build -t catalyst-classification:latest . -f docker/Dockerfile

docker-clean:
	rm -rf build/
	docker rmi -f catalyst-classification:latest
