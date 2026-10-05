"""Local (0-credit) evaluation: artifact structure, model statistics, ONNX Runtime measurements.

Heavy dependencies (torch, onnx, onnxruntime, numpy, psutil) are imported lazily and
only exist in the Python 3.11 environment. Every function accepts injected modules /
factories so the logic is unit-tested on Python 3.14 without them.
"""
