# src/permutations.py
from __future__ import annotations
import numpy as onp
import itertools as it

# conversion to int utils
def int_vec(v):
  return tuple(onp.rint(onp.asarray(v)).astype(int).tolist())

def int_mat(M):
  return onp.rint(onp.asarray(M)).astype(int)



def cycle_type(p):
  r""" cycle type from permutation `p` """
  seen = set()
  parts = []
  for i in range(len(p)):
    if i in seen:
      continue
    j = i
    k = 0
    while j not in seen:
      seen.add(j)
      j = p[j]
      k += 1
    parts.append(k)
  return tuple(sorted(parts, reverse=True))


def perm_hom_tup(items, image_fn):
  r""" permutation represented as tuple """
  lookup = {item: i for i, item in enumerate(items)}
  return tuple(lookup[image_fn(item)] for item in items)

def perm_hom(perm):
  r""" permutation representation homom """
  n = len(perm)
  M = onp.zeros((n, n), dtype=int)
  for j, i in enumerate(perm):
    # TODO why flipped?
    M[i, j] = 1
  return M


def compose_perm(p, q):
  return tuple(p[q[i]] for i in range(len(p)))

def perm_power(p, k):
  out = tuple(range(len(p)))
  for _ in range(k):
    out = compose_perm(p, out)
  return out


def sign_perm(p):
  inversions = 0
  for i in range(len(p)):
    for j in range(i + 1, len(p)):
      inversions += p[i] > p[j]
  return -1 if inversions % 2 else 1

