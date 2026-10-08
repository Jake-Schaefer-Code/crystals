r"""exact_gb.py — an EXACT (rational) Gröbner engine for the decision / structure phase.

PROVENANCE
----------
Vendored from ``agca_jax`` (``src/ideals/exact_gb.py``, Alex Gura). Changes from upstream: the
``LEX`` / ``GRLEX`` tags are defined here instead of imported from ``src.core.orders``; the
numpy-based ``border_coefficients`` and the converters from agca's float ``Polynomial`` /
``Ideal`` (``poly_from_library``, ``gens_from_ideal``, ``MAX_DEN``) are dropped, since they need
agca objects. The algorithms are unchanged. What follows is the upstream docstring, so mentions of
"the library" mean agca.

WHY THIS EXISTS
---------------
Ideal membership ("is 1 ∈ I?"), the quotient staircase, Krull dimension, and radical are
*decisions*.  The library's differentiable JAX engine decides them in float64 with
scale-blind zero tests (`simplify()` uses exact `==0`; `poly_div_single` pops leading-term
residue with an absolute `< 1e-9`).  On exactly-rational inputs a chain of S-polynomial
reductions drifts by roundoff until one residual is misclassified, a spurious low-degree
element enters the basis, and it cascades to a false `1 ∈ I`.  (Receipt: on
⟨x²+y²−1, θxy+y²−¼⟩ the float engine declares 1 ∈ I at 14/29 values of θ, while sympy over ℚ
returns a clean 3-element basis with the *same* rational coefficients the float engine was
approximating.)

You cannot make a *decision* robust with a tolerance — every fixed threshold has adversarial
inputs on both sides.  The fix is exact arithmetic in the decision path.  This module is a
compact Buchberger over ℚ (`fractions.Fraction`), used for the non-differentiable structure
phase only.  The float JAX engine is retained unchanged for the differentiable readouts,
which take the structure *as given* and never decide membership.

SCOPE / HONESTY
---------------
* Orders: LEX and GRLEX (covers `solve_border_basis`, the t-trick, and every Campaign-III
  path).  GRREVLEX raises — add it if a caller needs it.
* Inputs are exact: build polynomials with `poly_from_terms`, which takes `Fraction`/`int`/`str`
  coefficients. (Upstream also rationalised floats via `limit_denominator`; that path is dropped.)
* Small systems only — this is Python, not a compiled CAS; it is meant for the 2–4 variable,
  modest-degree regime the differentiable layer serves.
"""
from __future__ import annotations
from fractions import Fraction
import itertools


# Monomial-order tags. These match the integer constants agca uses, so term dicts and order tags
# are interchangeable with the upstream engine.
LEX = 0
GRLEX = 1

# A polynomial is a dict {exponent_tuple(int,...): Fraction}, zero coefficients pruned.
Poly = dict


# ── monomial order → total-order key (larger monomial ⇒ larger key) ───────────────
def _key(order_type, e):
    if order_type == GRLEX:
        return (sum(e),) + tuple(e)     # degree first, then lex on (x0, x1, …)
    if order_type == LEX:
        return tuple(e)                 # x0 most significant
    raise NotImplementedError(
        f"exact_gb supports LEX and GRLEX; got order_type={order_type}")


def lm(f, order):
    """Leading-monomial exponent tuple of a nonzero poly f under `order`."""
    return max(f.keys(), key=lambda e: _key(order, e))


# ── monomial + polynomial arithmetic (exact) ──────────────────────────────────────
def _divides(a, b):
    return all(x <= y for x, y in zip(a, b))


def _mono_div(b, a):                    # b / a, assumes a | b (so b_i >= a_i)
    return tuple(bi - ai for bi, ai in zip(b, a))


def _prune(f):
    return {e: c for e, c in f.items() if c != 0}


def _axpy(f, c, g, shift):
    """Return f + c · x^shift · g (all exact)."""
    out = dict(f)
    for e, v in g.items():
        ee = tuple(a + b for a, b in zip(e, shift))
        nv = out.get(ee, Fraction(0)) + c * v
        if nv == 0:
            out.pop(ee, None)
        else:
            out[ee] = nv
    return out


def _monic(f, order):
    lc = f[lm(f, order)]
    if lc == 1:
        return dict(f)
    return {e: v / lc for e, v in f.items()}


# ── S-polynomial and multivariate reduction ───────────────────────────────────────
def _spoly(f, g, order):
    lf, lg = lm(f, order), lm(g, order)
    L = tuple(max(a, b) for a, b in zip(lf, lg))
    s = _axpy({}, Fraction(1) / f[lf], f, _mono_div(L, lf))
    return _axpy(s, -Fraction(1) / g[lg], g, _mono_div(L, lg))


def reduce_poly(f, G, order):
    """Full multivariate remainder of f modulo the list G (exact). Since we only call
    this with an actual Gröbner basis for the structure readouts, the remainder is the
    canonical normal form and is supported on standard monomials."""
    f = _prune(dict(f))
    r = {}
    Glm = [(g, lm(g, order)) for g in G]
    while f:
        lf = lm(f, order)
        cf = f[lf]
        for g, lg in Glm:
            if _divides(lg, lf):
                f = _axpy(f, -cf / g[lg], g, _mono_div(lf, lg))
                break
        else:
            r[lf] = cf
            del f[lf]
    return r


# ── Buchberger + reduced Gröbner basis ─────────────────────────────────────────────
def buchberger(F, order):
    """Reduced, monic Gröbner basis (over ℚ) of the ideal generated by polys F."""
    G = [_prune(dict(f)) for f in F]
    G = [g for g in G if g]
    pairs = [(i, j) for i in range(len(G)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        gi, gj = G[i], G[j]
        li, lj = lm(gi, order), lm(gj, order)
        if all(min(a, b) == 0 for a, b in zip(li, lj)):   # product (coprime-LT) criterion
            continue
        r = reduce_poly(_spoly(gi, gj, order), G, order)
        if r:                                              # nonzero remainder ⇒ new element
            G.append(r)
            k = len(G) - 1
            pairs.extend((k, m) for m in range(k))
    return _reduce_gb(G, order)


def _reduce_gb(G, order):
    """Minimalise (drop LT-redundant), inter-reduce, and make monic.

    Minimalisation keeps exactly ONE generator per distinct leading monomial: process
    generators smallest-LM-first and keep g only if no ALREADY-KEPT generator's LM
    divides LM(g).  (A naive 'drop g if any other LM divides it' deletes BOTH members of
    an equal-LM pair — which silently drops the pure-power generator that certifies
    zero-dimensionality.)"""
    G = [_monic(g, order) for g in G]
    G = sorted(G, key=lambda g: _key(order, lm(g, order)))          # ascending by LM
    kept = []
    for g in G:
        lg = lm(g, order)
        if not any(_divides(lm(h, order), lg) for h in kept):
            kept.append(g)
    out = []
    for i, g in enumerate(kept):
        others = [h for k, h in enumerate(kept) if k != i]
        out.append(_monic(reduce_poly(g, others, order), order))   # LM survives (minimal)
    return out


# ── decisions & structure built on the exact GB ───────────────────────────────────
def is_one_in_ideal(gb, n_vars):
    """True iff 1 ∈ I, i.e. the reduced GB is {1} — the single generator IS a nonzero
    constant.  Test that the generator is supported *only* on the constant monomial, not
    merely that it *has* a constant term: a principal ideal like ⟨x − 1/2⟩ or ⟨x² − 1⟩ has a
    one-element reduced GB whose generator carries a nonzero constant term yet is NOT the unit
    ideal.  The earlier `zero in gb[0]` misclassified every such ideal as 1 ∈ I (and so made
    `standard_monomials` wrongly report R/I = 0 for any principal ideal with a constant term)."""
    zero = tuple([0] * n_vars)
    return len(gb) == 1 and set(gb[0].keys()) == {zero}


def standard_monomials(gb, n_vars, order=GRLEX):
    """The quotient K-basis B = monomials NOT divisible by any leading monomial of the GB.
    Raises ValueError if R/I is infinite-dimensional (matches QuotientModule)."""
    if is_one_in_ideal(gb, n_vars):
        return []                                     # R/I = 0
    leads = [lm(g, order) for g in gb]
    # zero-dimensional ⇔ every variable has a pure-power leading term.
    bound = [None] * n_vars
    for L in leads:
        nz = [k for k, e in enumerate(L) if e > 0]
        if len(nz) == 1:
            k = nz[0]
            bound[k] = L[k] if bound[k] is None else min(bound[k], L[k])
    if any(b is None for b in bound):
        raise ValueError("R/I is infinite-dimensional (not zero-dimensional); the exact "
                         "quotient basis is only defined for zero-dimensional ideals.")
    B = []
    for exps in itertools.product(*[range(b) for b in bound]):
        if not any(_divides(L, exps) for L in leads):
            B.append(tuple(exps))
    return B


def poly_from_terms(terms, n_vars):
    """Build an exact Poly from {exponent_tuple: Fraction|int|str}. For exact-input callers
    that want to bypass float rationalisation entirely."""
    return _prune({tuple(int(v) for v in e): Fraction(c) for e, c in terms.items()})


# ── membership & saturation (exact decisions via the classical t-tricks) ───────────
def ideal_membership(gb, f, order):
    """Decide plain ideal membership  f ∈ I  exactly.

    Identity:  f ∈ I  ⟺  the multivariate remainder of f modulo a Gröbner basis of I is 0.
    (Normal-form membership; the remainder is well defined — order-independent as *zero or
    not* — precisely because `gb` is a Gröbner basis.)

    PRECONDITION: `gb` must already be a Gröbner basis of I under `order` (e.g. the output of
    `buchberger(F, order)`).  Passing an arbitrary generating set gives a WRONG answer — a
    nonzero remainder against a non-GB does not certify non-membership.
    """
    return not reduce_poly(f, gb, order)


def _embed_t(gens, extra, n):
    """Embed I = ⟨gens⟩ and the t-trick generator  1 − t·extra  into k[t, x_1…x_n], with the
    auxiliary variable t placed at exponent index 0 (prepended coordinate).  Returns the
    augmented generator list over n+1 variables.  Shared by `radical_membership` and
    `saturate` so the embedding is written once."""
    embedded = [{(0,) + e: v for e, v in g.items()} for g in gens]
    tgen = {(0,) * (n + 1): Fraction(1)}                 # the constant 1
    for e, v in extra.items():                           # minus t·extra  (t-exponent 1)
        tgen[(1,) + e] = -v
    embedded.append(_prune(tgen))
    return embedded


def _infer_n_vars(gens, extra):
    """Number of variables of the original ring, read off any nonzero exponent tuple."""
    for f in list(gens) + [extra]:
        for e in f:
            return len(e)
    raise ValueError("cannot infer the number of variables from all-zero polynomials.")


def radical_membership(gens, c, order=GRLEX):
    """Decide radical membership  c ∈ √I  exactly, where I = ⟨gens⟩ (Rabinowitsch trick).

    Identity (Rabinowitsch / weak Nullstellensatz):
        c ∈ √I   ⟺   1 ∈ I + ⟨1 − t·c⟩   in   k[t, x_1…x_n].
    We adjoin t as coordinate index 0, embed each generator of I, add 1 − t·c, compute a
    Gröbner basis, and test 1 ∈ J.  Whether 1 lies in an ideal is order-independent, so the
    cheaper GRLEX is the default (no elimination order is needed here — contrast `saturate`).

    This decides *membership in the radical* only; it does NOT compute √I itself and needs no
    primary decomposition or factorisation.

    SCOPE: exact over ℚ, small systems (2–4 vars, modest degree), LEX/GRLEX only — the same
    honest envelope as the rest of this module.
    """
    if not c:                                            # 0 ∈ √I for every ideal I
        return True
    n = _infer_n_vars(gens, c)
    J = _embed_t(gens, c, n)
    gb = buchberger(J, order)
    return is_one_in_ideal(gb, n + 1)


def saturate(gens, f, order=LEX):
    """Compute the saturation  I : f^∞  exactly, where I = ⟨gens⟩ (elimination t-trick).

    Identity:
        I : f^∞  =  (I + ⟨1 − t·f⟩) ∩ k[x_1…x_n]      (eliminate the adjoined variable t).
    We adjoin t as coordinate index 0, embed I, add 1 − t·f, compute a Gröbner basis under an
    ELIMINATION order for t, and keep the t-free generators (Elimination Theorem: G ∩ k[x] is
    a Gröbner basis of J ∩ k[x]).  LEX with t most significant *is* such an order — and since t
    is coordinate 0, plain LEX on the (1+n)-tuples eliminates it.  The kept generators are then
    stripped of their (always-zero) t-coordinate and returned as `Poly`s over the original n
    variables.

    Returns a reduced Gröbner basis of I : f^∞ under LEX.  If the saturation is the whole ring
    it returns ⟨1⟩ = [poly_from_terms({(0,)*n: 1}, n)]; if it is the zero ideal it returns [].

    PRECONDITION: `order` must be a global elimination order for t.  LEX (the default) is the
    only supported choice — GRLEX does NOT eliminate t and would give a wrong answer.

    SCOPE: exact over ℚ, small systems, LEX only.  Radical *membership* (not the radical
    ideal) plus this saturation are all the decision phase needs — no decomposition.
    """
    n = _infer_n_vars(gens, f)
    J = _embed_t(gens, f, n)
    gb = buchberger(J, order)
    if is_one_in_ideal(gb, n + 1):                       # I : f^∞ = whole ring
        return [poly_from_terms({(0,) * n: 1}, n)]
    kept = [g for g in gb if all(e[0] == 0 for e in g)]  # t-free ⇒ in the elimination ideal
    return [_prune({e[1:]: v for e, v in g.items()}) for g in kept]
