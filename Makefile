.PHONY: classification

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	bash ./bin/_check_codestyle.sh

classification: requirements.txt
	docker build -t catalyst-classification:latest . -f docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-classification:latest
