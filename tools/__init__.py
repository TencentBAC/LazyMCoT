"""Tools — Conformal Safe-Skip Router (training-free, no hard-threshold)."""
from .router import Router, question_priors

# Legacy alias for backward compatibility
BinaryConservativeRouter = Router

__all__ = ["Router", "BinaryConservativeRouter", "question_priors"]
