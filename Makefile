.PHONY: help install test preflight eval-stage1 eval-stage2 benchmark clean run-pi run-pi-web bench-pi

PYTHON ?= python3

help:
	@echo "Available commands:"
	@echo "  make install       - Install required Python dependencies"
	@echo "  make test          - Run automated unit tests"
	@echo "  make preflight     - Run Stage 2 prerequisite preflight verification"
	@echo "  make eval-stage1   - Evaluate Stage 1 FP32 model on frozen test split"
	@echo "  make eval-stage2   - Compare FP32 vs INT8 models on test and OOD splits"
	@echo "  make benchmark     - Run Gate 5 hardware latency & memory benchmark"
	@echo "  make bench-pi      - Run standalone ONNX CPU benchmark (FP32 vs INT8 & thread sweep)"
	@echo "  make run-pi        - Run webcam face detection using the Pi package"
	@echo "  make run-pi-web    - Run Flask MJPEG web streaming face detection server"
	@echo "  make clean         - Remove bytecode, cache, and temporary build files"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover tests

preflight:
	$(PYTHON) scripts/stage2_preflight.py --config configs/stage2.json

eval-stage1:
	$(PYTHON) scripts/evaluate.py --weights models/stage1/best.pt --name E1 --imgsz 416

eval-stage2:
	$(PYTHON) scripts/evaluate_stage2.py --config configs/stage2.json

benchmark:
	$(PYTHON) scripts/bench_target.py --config configs/stage2.json

run-pi:
	$(PYTHON) pi/detect_webcam.py

run-pi-web:
	$(PYTHON) pi/app_web.py

bench-pi:
	$(PYTHON) pi/bench_pi.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	find . -type f -name "*.cache" -delete 2>/dev/null || true
	rm -rf .pytest_cache/
