"""Numerical checks for the three raw MDV6 detection scales (no NPU access)."""
import torch


def compare_detection_outputs(reference, actual, class_tolerance, vector_tolerance):
    if len(reference) != 3 or len(actual) != 3:
        raise ValueError('Expected exactly three detection scales')
    for expected, observed in zip(reference, actual):
        if len(expected) != 3 or len(observed) != 3:
            raise ValueError('Expected class, anchor, and vector tensors per scale')
        if any(a.shape != b.shape for a, b in zip(expected, observed)):
            raise ValueError('Detection tensor shapes differ')
    finite = all(torch.isfinite(t).all().item() for outputs in (reference, actual)
                 for scale in outputs for t in scale)
    # JSON reports use null rather than non-standard NaN/Infinity values.
    max_cls = max((a[0].float() - b[0].float()).abs().max().item()
                  for a, b in zip(reference, actual)) if finite else None
    max_vec = max((a[2].float() - b[2].float()).abs().max().item()
                  for a, b in zip(reference, actual)) if finite else None
    return dict(finite=finite, max_class_diff=max_cls, max_vector_diff=max_vec,
                ok=finite and max_cls < class_tolerance and max_vec < vector_tolerance)
