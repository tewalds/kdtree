.PHONY: clean run_benchmark run_test

run_test:
	mkdir -p build
	cmake -S . -B build
	cmake --build build
	build/kdtree_test --skip-benchmarks
	. venv/bin/activate && pip install -e . && python3 -m pytest kdtree_test.py

run_benchmark:
	. venv/bin/activate && pip install -e . && ./bench_runner.py

clean:
	rm -rf \
		build/* \
		benchmarks/*_bin \
		plots/* \
		*.o \
		*.so \
		*.egg-info \
		.pytest_cache \
		__pycache__
