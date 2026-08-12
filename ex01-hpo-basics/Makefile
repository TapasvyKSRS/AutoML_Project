help:
	@echo "Makefile help: (Tested on Linunx)"
	@echo "* install	to install the requirements into your current virtaul env"
	@echo "* test		to run the tests"
	@echo "* check 		to run the code style checker"

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests

check:
	pycodestyle --max-line-length=120 src
	@echo "All good!"

clean-tex:
	rm *.log *.aux *.out

tex:
	pdflatex *.tex
	make clean-tex


.PHONY: install test check all clean-tex tex help
.DEFAULT_GOAL := help
