import numpy as np
import scipy.linalg as la

# --- Define parameters for each sector ---
# We assume three sectors: (S,lambda)
# Sector 1: S = 0, lambda = +, multiplicity = 1
# Sector 2: S = 1, lambda = +, multiplicity = 2
# Sector 3: S = 1, lambda = -, multiplicity = 1

# Define a dictionary with keys (S, lambda) and values:
#   multiplicity dimension, J, K (coupling constants for f)
sectors = {
    (0, '+'): {'mult_dim': 1, 'J': 0.0, 'K': 0.0},   # For S=0, S(S+1)=0 so f=0
    (1, '+'): {'mult_dim': 2, 'J': 1.2, 'K': 0.3},     # For S=1, S(S+1)=2, and O_pt=+1
    (1, '-'): {'mult_dim': 1, 'J': 1.2, 'K': -0.3},    # For S=1, S(S+1)=2, and O_pt=-1
}

# For each sector, the canonical invariant value is:
#   S(S+1) and O_pt is +1 for '+' and -1 for '-' sectors.
def f_invariant(S, lam, J, K):
    # f(S(S+1), O_pt) = J * S(S+1) + K * O_pt
    O_pt = 1 if lam == '+' else -1
    return J * (S*(S+1)) + K * O_pt

# We'll now construct each block of the Hamiltonian using the ansatz:
#   H_sector = f(S(S+1), O_pt) * I_canonical  \otimes  H_mult
# For an irreducible (canonical) factor, by Schur's lemma the operator is proportional to the identity.
# The extra freedom appears in the multiplicity space, and we will choose an arbitrary Hermitian matrix for that.

blocks = []
sector_labels = []  # For reference

# For reproducibility, set random seed
np.random.seed(123)

# Loop over sectors
for (S, lam), params in sectors.items():
    mult_dim = params['mult_dim']
    J_val = params['J']
    K_val = params['K']
    
    # Compute the canonical (invariant) scalar f = f(S(S+1), O_pt)
    f_val = f_invariant(S, lam, J_val, K_val)
    
    # In the canonical factor, the operator is f_val * I (I of dimension d_can).
    # Since the representation is irreducible, we assume d_can = 1 (the operator acts as a scalar).
    # (In a more realistic situation, d_can > 1 but by Schur's lemma the invariant operator is f_val * I.)
    canonical_part = np.array([[f_val]])
    
    # For the multiplicity part, create an arbitrary Hermitian matrix of size mult_dim x mult_dim.
    # For multiplicity 1, this is just a number.
    A = np.random.randn(mult_dim, mult_dim)
    H_mult = (A + A.T) / 2  # enforce Hermiticity
    
    # The sector Hamiltonian block is given by the Kronecker product:
    block = np.kron(canonical_part, H_mult)
    blocks.append(block)
    
    sector_labels.append((S, lam))

# Construct the full Hamiltonian as a block-diagonal matrix:
H_invariant = la.block_diag(*blocks)

print("Hamiltonian from the invariant formulation:")
print(H_invariant)
print("\nSector labels (S, lambda):", sector_labels)

# Now, let us simulate the projection method.
# In a projection-based approach, you would build projectors P_(S,lambda)
# that select each block from the full Hilbert space.
# Here, we already have the block-diagonal structure.
# So, define H_projection as exactly H_invariant.
H_projection = H_invariant.copy()

# Diagonalize both Hamiltonians:
eig_invariant, _ = la.eig(H_invariant)
eig_projection, _ = la.eig(H_projection)

print("\nEigenvalues (Invariant formulation):", np.sort(eig_invariant.real))
print("Eigenvalues (Projection method):", np.sort(eig_projection.real))

# --- Comparison and Discussion ---
#
# In this example, both formulations yield exactly the same block-diagonal Hamiltonian.
# However, note that in our "invariant formulation" we explicitly parameterized the continuous
# (spin) part as f(S(S+1), O_pt) = J * S(S+1) + K * O_pt.
#
# For a realistic system (say, in a PHF calculation), one would not assume the form of f a priori.
# Instead, f would be determined either by microscopic derivations, effective field theory, or fitted
# to experimental data. The benefit of our formulation is that it reduces the number of free parameters:
# instead of having an arbitrary block for each (S, lambda), one only needs to specify the (few) parameters
# J and K (and possibly additional ones if higher-order terms are needed) plus a smaller matrix for the multiplicity.
#
# In contrast, the standard projection method H = ⨁ P_(S,λ) H P_(S,λ) guarantees block-diagonality,
# but the structure inside each block is not made explicit.



