# Referee report: *Closed-form Optimal Prior*

## Recommendation

**Major revision before submission.**

The central optimization result is elegant and correct. The paper is potentially publishable, but the current draft overclaims strictness and scope, contains two sign errors in the Ising derivation, and draws DFA conclusions that are not supported over the full plotted parameter range.

## What checks out

Let

$$
p_t=G^t p_0,
\qquad
\bar q=\frac1N\sum_{t=0}^{N-1}p_t.
$$

Direct expansion gives, for every admissible prior $q$,

$$
M_N(q)-M_N(\bar q)
=N\left[D(\bar q\Vert q)-D(G\bar q\Vert Gq)\right]\geq 0.
$$

Therefore $\bar q$ is a global minimizer. I also checked the entropy form, Jensen--Shannon contraction interpretation, weighted extension, and Cesàro/stationary asymptotics under the paper's finite-state assumptions.

I evaluated the main identity on 160 randomly generated stochastic maps and distributions. The maximum absolute numerical error was $9.77\times10^{-15}$.

## Major findings

### 1. Strictness is false as stated (pages 1, 3, and 6--7)

The paper repeatedly suggests that the optimized bound is strictly positive whenever $G$ is nontrivial or is not a permutation. This is false.

First, $N=1$ gives $\bar q=p_0$, and therefore

$$
M_1^*=0
$$

for every $G$ and every $p_0$.

There are also counterexamples for every $N$. Consider a three-state deterministic map with columns

$$
0\mapsto1,\qquad 1\mapsto0,\qquad 2\mapsto0,
$$

and take $p_0=\delta_0$. The map is not a permutation of the full state space and $p_0$ is not fixed. Nevertheless, it is reversible on the two-state support visited by the trajectory, so $M_N^*=0$ for every $N$.

The exact statement is

$$
M_N^*
=\sum_{t=0}^{N-1}
\left[D(p_t\Vert\bar q)-D(Gp_t\Vert G\bar q)\right]\geq0.
$$

It is strictly positive exactly when at least one of these data-processing inequalities is strict. Equality holds exactly when $G$ is sufficient or recoverable for all relevant pairs $(p_t,\bar q)$.

This correction affects the title, abstract, theorem discussion, and conclusion. A safer title would be:

> A prior-independent entropy-production bound for repeated stochastic maps

### 2. The uniqueness condition is not correct (pages 3 and 10)

Ordinary Markov-chain irreducibility does not imply that $\bar q$ is the unique minimizer. An irreducible cyclic permutation is an immediate counterexample.

The complete minimizer set is characterized by
$$

\left\{q:
D(\bar q\Vert q)=D(G\bar q\Vert Gq)
\right\},

$$
subject to the usual support conditions. Uniqueness requires a strict data-processing or channel-sufficiency condition on the relevant support, not ordinary irreducibility alone.

### 3. Equation 10 compares incompatible quantities (page 3)

The displayed inequality compares the single-cycle cost $\mathcal C(p_0)$ with the accumulated $N$-cycle mismatch. Define

$$
\mathcal C_N(p_0):=
\sum_{t=0}^{N-1}\mathcal C(p_t)
$$

and write
$$
\mathcal C_N(p_0)
\geq M_N(q)
\geq M_N(\bar q)
\geq0.
$$
### 4. Two Ising signs are reversed (page 4, equations 17--18)

For
$$

H(\sigma)=-\frac{J}{2N}\sum_{i\ne j}\sigma_i\sigma_j,

$$
flipping spin $i$ gives

$$
\Delta H
=H(\sigma^i)-H(\sigma)
=\frac{2J}{N}\sigma_i\sum_{j\ne i}\sigma_j.
$$

Local detailed balance is

$$
\frac{K_\nu(\sigma^i\mid\sigma)}
{K_\nu(\sigma\mid\sigma^i)}
=\exp(-\beta_\nu\Delta H).
$$

The signs printed in equations 17 and 18 are reversed. Equation 19 and the repository code already use the correct convention, so these appear to be manuscript errors rather than errors in the numerical experiment.

### 5. The speed-limit function is wrong (page 5, equation 20)

Replace $\arctan$ with $\operatorname{arctanh}$. This agrees with the cited primary result and with the notebook implementation.

The two placeholders reading `Appendix BA: ADD` and `Appendix BA: add` must also be filled or removed.

### 6. The DFA conclusions overstate the computed data (pages 6 and 15)

I reconstructed the D3, D5, D7, and D9 automata and checked language equivalence on every binary string of length 0 through 12: 8,191 inputs in total. All four machines passed this test.

Their thermodynamic cost ordering is not universal, however. For $p(0)=0.9$ and $N=3$, I obtain

- D3: 0.09230
- D5: 0.28634
- D7: 0.40941
- D9: 0.36947

Thus D9 is below D7, contradicting monotonic scaling with the number of extra states at this horizon.

There are also short-horizon cases where the minimal DFA is not best. For example, at $p(0)=0.1$ and $N=3$,

- D3: 0.44128
- D5: 0.36213

Across the plotted $p(0)$ values, I found four cases in which D3 was not the minimum: $p(0)=0.1$ at $N=3,4,5$, and $p(0)=0.2$ at $N=3$.

The longer-horizon trend remains convincing. The paper should describe the observed regime precisely rather than claim universal minimality. If a universal result is intended, it requires a separate theorem with explicit assumptions.

### 7. “Any periodic process” is too broad

The theorem applies cleanly to a finite-state, fixed linear stochastic map $G$ whose repeated-cycle cost admits the same prior and a nonnegative residual contribution.

It does not automatically cover a system-only process with memory for which no fixed linear map closes the dynamics. The phrase “$G$ need not be Markovian” is also misleading: $G$ is itself a stochastic transition kernel. A better statement is that it need not be embeddable in a continuous-time Markov chain, while intra-cycle dynamics may require enlargement of the state space.

The Ising example uses a time-independent generator. Such a generator is autonomous and has every time interval as a formal period. The manuscript should call $\Delta t=0.1$ a sampling interval, or replace the example with genuinely time-periodic driving if periodicity is meant physically.

## Independent numerical audit

Using the notebook parameters of nine spins, $J=0.2$, bath inverse temperatures $(3,1)$, rates $(1,1)$, $\Delta t=0.1$, and the all-up initial state, I independently evaluated the propagator using uniformization.

Results:

- Local-detailed-balance maximum log-ratio error: $2.22\times10^{-16}$.
- Entropy-production decomposition maximum error for $N=1,\ldots,200$: $1.01\times10^{-10}$.
- Minimum value of total EP minus the optimized mismatch bound: 2.3346.
- Maximum fraction $M_N^*/\mathrm{EP}_N$: 0.5946 at $N=59$.
- Stationary limit $D(p_0\Vert\pi)$: 4.68299.

These results support the qualitative claims made for the spin experiment, including the claim that the optimized bound can exceed one half of total entropy production.

## Interpretation issues

The manuscript should avoid calling $M_N^*$ the “minimum entropy production” without qualification. It minimizes the mismatch component over mathematical priors and provides a prior-independent lower bound. Physical realizability of the optimizing prior and minimization of the residual contribution are separate questions.

When $\pi$ is a Gibbs equilibrium distribution, $D(p_0\Vert\pi)$ is dimensionless. The corresponding excess free energy is

$$
k_BT D(p_0\Vert\pi)=\beta^{-1}D(p_0\Vert\pi).
$$

The asymptotic discussion should include this factor rather than identifying the divergence itself with a dimensional free energy.

The statement that the prior encodes “everything thermodynamically relevant” is also too strong because the residual or baseline contribution remains important.

## Reproducibility recommendations

The numerical story is largely reproducible, but the current notebooks are not yet a reliable archival workflow.

Specific issues found:

- The Ising notebook computes an exact Wasserstein distance and then overwrites it with an approximate optimizer result.
- Because the initial distribution is a delta distribution and the ground metric is Hamming distance, this Wasserstein distance has a direct closed form and does not require numerical optimization.
- The active Figure 5 DFA source cell uses $p=0.3$, while the stored output and manuscript caption correspond to $p=0.1$.
- One plotting cell contains an absolute path in a coauthor's home directory.
- Several results depend on stale notebook execution order.

Before submission, I recommend:

1. Create one deterministic script for each paper figure.
2. Save the numerical arrays behind every figure to CSV or NPZ.
3. Add assertions for stochasticity, normalization, local detailed balance, the main identity, and DFA language equivalence.
4. Rebuild every figure in a clean process rather than relying on notebook outputs.
5. Record exact dependency versions and random seeds.
6. Use relative output paths.

The complexity footnote should also be reconsidered. Dense trajectory propagation by repeated matrix--vector multiplication is $O(Nd^2)$, not $O(Nd^3)$. Sparse deterministic maps can be cheaper still.

## Editorial corrections

- Replace the Figure 6 caption, which currently reads only “Caption.”
- Remove the stray horizontal rule and excess whitespace on page 14.
- Correct “Jenson--Shannon,” “Currie--Weiss,” “nonequilibirum,” repeated words, article errors, and similar typographical problems.
- State clearly in the main DFA discussion that the transition map describes an ensemble driven by independently sampled input symbols.
- Tighten the references/layout so the final reference fragment does not occupy a nearly empty sixteenth page.

## Submission-critical checklist

- [ ] Rewrite the title, abstract, and discussion so strictness is conditional.
- [ ] State the exact equality and minimizer sets.
- [ ] Correct equations 10, 17, 18, and 20.
- [ ] Qualify the DFA claims and regenerate its plots from a clean run.
- [ ] Clarify the stochastic-map and periodic-process scope.
- [ ] Convert the notebooks into deterministic figure-generation scripts.
- [ ] Replace all placeholders and incomplete captions.
- [ ] Complete a professional copy-editing pass.
