.PHONY: classification

classification: requirements.txt
	docker build -t catalyst-classification:latest . -f docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-classification:latest
