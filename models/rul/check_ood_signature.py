"""
check_ood_signature.py

Inspect compute_ood_zscore_distance's real signature and docstring before
calling it -- avoid a third wrong guess in a row.
"""
import inspect
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul_ensemble import compute_ood_zscore_distance

print("Signature:")
print(" ", inspect.signature(compute_ood_zscore_distance))
print("\nDocstring:")
print(inspect.getdoc(compute_ood_zscore_distance))
print("\nSource:")
print(inspect.getsource(compute_ood_zscore_distance))