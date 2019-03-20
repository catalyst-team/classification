.PHONY: finetune

finetune: requirements.txt
	docker build -t catalyst-finetune:latest . -f docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-finetune:latest
