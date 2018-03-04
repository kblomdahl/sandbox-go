.PHONY: build clean init run

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

run: build
	python -m sandbox_go
