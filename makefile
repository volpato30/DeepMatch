A=128
F=256
H=4
init_lr=0.1

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
	python main.py  --A $(A) --F $(F) --H $(H) --init_lr $(init_lr) --optimizer momentum --activation relu --mode train

.PHONY: prep
prep:
	python main.py --mode prep

.PHONY: infer
infer:
	python main.py --mode infer
