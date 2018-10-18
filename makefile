A=128
F=256
H=4

.PHONY: clean
clean:
	rm -rf build
	rm -f cython_func.c
	rm -f cython_func*.so

.PHONY: build
build: clean
	python cython_setup.py build_ext --inplace

.PHONY: test
test: build
	rm -f test_data/test_scans.tfrecord
	python unit_test.py

.PHONY: train
train:
	rm -rf chkpoint/*
	python main.py --mode train --A $(A) --F $(F) --H $(H)

.PHONY: prep
prep:
	rm -rf data/*.tfrecord
	python main.py --mode prep

