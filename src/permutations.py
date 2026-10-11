# src/permutations.py
"""Compatibility alias: ``src.permutations`` moved to ``src.symmetry.permutations``.

Several notebooks (``notebooks/rep``, ``notebooks/quantum``) and ``src.quantum_symmetry`` still import
the pre-restructure flat path. New code should import ``src.symmetry.permutations``.
"""
from src.symmetry.permutations import *  # noqa: F401,F403
