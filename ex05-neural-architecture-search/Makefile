.PHONY: install test check all clean-tex tex help run
.DEFAULT_GOAL := help

help:
	@echo "Makefile help: (Tested on Linux)"
	@echo "* install	to install the requirements into your current virtual env"
	@echo "* run		to run NAS"
	@echo "* test		to run the tests"
	@echo "* check 	to run the code style checker"

install:
	python -m pip install -r requirements.txt
	@echo "Getting the benchmark ----------------------------------"
	wget --no-clobber https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

test: run
	python -m pytest tests

run:
	python run.py

check:
	pycodestyle --max-line-length=120 src
	@echo "All good!"

clean-tex:
	rm *.log *.aux *.out

tex:
	pdflatex *.tex
	make clean-tex
