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
	python unit_test.py
