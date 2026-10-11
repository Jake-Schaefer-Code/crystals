# tests/test_compat_shims.py
import src.permutations as old
import src.symmetry.permutations as new


def test_old_permutations_path_re_exports_the_moved_module():
  for name in ("generated_group", "from_cycles", "sgn", "PermutationGroup", "Permutation"):
    assert getattr(old, name) is getattr(new, name)


def test_quantum_symmetry_imports_through_the_old_path():
  import src.quantum_symmetry  # noqa: F401  (imports ``src.permutations``)
