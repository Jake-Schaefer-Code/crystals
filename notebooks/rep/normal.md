# Open Quantum Final: What Should Survive Into Code

Only a small part of the paper should survive as actual implementation guidance.

## Core Pipeline

1. Build a symmetry representation `U_g` on Hilbert space `H`.
2. Lift it to operator space as `R_g = U_g^* \otimes U_g` on `End(H)`.
3. Check covariance of the dynamics:
   - Hamiltonian case: `[H, U_g] = 0`
   - Open-system case: `[L_super, R_g] = 0`
4. Use characters or projectors to decompose `H` or `End(H)` into invariant sectors.
5. Restrict the Hamiltonian or Lindbladian to each sector and study them separately.
6. Identify fixed sectors, dark states, commutants, or noiseless factors.

## What Is Computationally Useful

- `Representation.projector(...)` in [src/symmetry.py](/Users/jakeschaefer/Desktop/Research_Stuff/mat_sci/crystals/src/symmetry.py:144)
- `reynolds_projector()` for invariant pieces in [src/symmetry.py](/Users/jakeschaefer/Desktop/Research_Stuff/mat_sci/crystals/src/symmetry.py:165)
- Liouville lift, commutants, and Lindblad helpers in [src/quantum_symmetry.py](/Users/jakeschaefer/Desktop/Research_Stuff/mat_sci/crystals/src/quantum_symmetry.py:44)

## What Not To Keep

- The long philosophy-of-probability framing is not helping the code.
- The paper should not blur Hilbert-space symmetry sectors with operator-space sectors.
- The phrase "the Lindbladian is governed by the exponential map of the Kraus operators" is too loose to use as a coding principle.

## Practical Rule

If the object of interest is a density matrix or channel, work in Liouville space first.  
If the object of interest is a state vector or Hamiltonian, work on Hilbert space first.

That distinction is the main thing the paper needs to preserve if it is going to support actual computations.
