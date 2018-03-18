.PHONY: build clean init run

DEVICE?=0

all: build

build: init
	python setup.py build_ext --inplace

clean:
	find sandbox_go/ -type f -name '*.c' -delete
	find sandbox_go/ -type f -name '*.so' -delete
	find sandbox_go/ -type d -name '__pycache__' -prune -exec rm -rf '{}' ';'
	rm -rf build/

init:
	pip install -r requirements.txt

docker:
	docker build -t "sandbox_go/sandbox_go:0.0.1" .
	docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(DEVICE) \
		-v "$(realpath data/):/app/data" \
		-v "$(realpath models/):/app/models" \
		"sandbox_go/sandbox_go:0.0.1"
