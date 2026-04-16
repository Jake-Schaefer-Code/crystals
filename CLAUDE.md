# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- `pytest -s -vv` - Run all tests with verbose output
- `pytest jaxqmc/tests/test_mcmc_position.py` - Run specific test file
- `pytest -k "test_gaussian_basic"` - Run tests matching pattern
- Use poetry for pytest.
- **IMPORTANT**: Use poetry.

### Type Checking
- `mypy jaxqmc` - Run static type checking on the main package

## Project Architecture

This is a JAX-based Quantum Monte Carlo (QMC) implementation for electronic structure calculations. The project implements variational Monte Carlo methods with focus on Slater determinant wavefunctions.

### Key Components

**Core Structure:**
- `jaxqmc/` - Main package containing all QMC functionality
- `examples/` - Example scripts (e.g., H2 molecule calculations)
- `conf/` - Hydra configuration files for different systems and basis sets
- `scratch/` - Development/experimental scripts

**Main Modules:**
- `energy.py` - Local energy calculation (`construct_local_energy`)
- `mcmc/sampler.py` - Composite MCMC sampler with finite state machine architecture
- `wavefunctions/slater.py` - Slater determinant wavefunction implementation
- `atomic.py` & `electrons.py` - Atomic and electronic structure utilities
- `ewald.py` - Ewald summation for periodic systems
- `hartree_fock/` - Hartree-Fock integration (uses PySCF)

**MCMC Sampling:**
The sampling system uses a finite state machine approach with multiple subsampler types:
- `RandomDirectionSliceSampler` - Slice sampling in random directions
- `BoxSliceSampler` - Box-constrained slice sampling
- `BoxSingleSliceSampler` - Single-electron box slice sampling

**Types:**
- `ElectronicState` - Dataclass containing electron positions and spins
- `PRNGKey` & `LogPDF` - Type aliases for JAX random keys and log probability functions

### Configuration System

Uses Hydra for configuration management with YAML files in `conf/`:
- `system/` - Molecular systems (H2.yaml, NaCl.yaml)
- `ansatz/` - Basis set configurations (sto-3g.yaml, bse-default.yaml)
- `finite/` - Finite system parameters

### Dependencies

Key external dependencies:
- JAX/JAXlib for automatic differentiation and JIT compilation
- PySCF for Hartree-Fock calculations and basis sets
- PyQMC for comparison/validation
- Chex for dataclasses and testing utilities
- JAXtyping for type annotations

### Example Usage Pattern

See `examples/h2.py` for typical workflow:
1. Define molecular system using `Molecule` class
2. Set up basis set with `BSEMolecularBasisSet`
3. Create Slater wavefunction with `SlaterWF`
4. Construct local energy function with `construct_local_energy`
5. Build composite sampler with multiple MCMC methods
6. Run sampling and energy evaluation loops

## VMC Configuration System

### Configuration-Driven Architecture

The codebase uses a configuration-driven architecture where **the presence or absence of components determines the workflow**:

**Core Principle**: No `optimizer` in config = single energy calculation (not optimization)

**Entry Point**: `python -m jaxqmc.run +study=config_name`

### Configuration Structure

#### Energy Calculation Mode (no optimizer)
```yaml
defaults:
  - /system: H2           # Molecular/crystal structure
  - /ansatz: slater       # Wavefunction ansatz
  - /sampler: metropolis  # MCMC sampling method
  - _self_

# Unified initialization approach
initialization:
  method: hartree_fock    # or: random, from_file
  file: null             # for from_file method

# Sampling configuration
sampler:
  step_size: 1.0
  num_walkers: 200
  num_steps: 20000       # Total samples for energy estimate
  burn_in: 2000

# Simple output specification
outputs:
  energy: energy.txt
  samples: trajectories.h5
  analysis: blocking.json

seed: 42
```

#### Optimization Mode (with optimizer)
```yaml
defaults:
  - /system: H2
  - /ansatz: slater
  - /sampler: metropolis
  - /optimizer: adam      # Presence triggers optimization mode
  - _self_

initialization:
  method: hartree_fock    # or: random, from_checkpoint
  checkpoint: null        # for from_checkpoint method

sampler:
  step_size: 1.0
  num_walkers: 200
  num_steps: 1000        # Samples per optimization iteration
  burn_in: 500

# Optimizer section - its presence changes workflow
optimizer:
  learning_rate: 0.01
  max_iterations: 5000
  
  convergence:
    tolerance: 1e-6
    check_every: 100
    window_size: 1000
    
  # Checkpointing lives inside optimizer (semantic clarity)
  checkpointing:
    enabled: true
    frequency: 100       # Every N optimization steps
    filename: checkpoint.h5
    keep_best: true
    max_files: 3

outputs:
  convergence: convergence.pdf
  final_params: parameters.json
  history: optimization.h5

seed: 42
```

### Key Semantic Differences

**`sampler.num_steps` interpretation**:
- Without optimizer: Total number of samples for statistical energy estimate
- With optimizer: Number of samples per optimization iteration

**Checkpointing behavior**:
- Lives inside `optimizer` config (only relevant during optimization)
- `frequency` is in optimization steps, not individual samples
- For energy calculations, use `outputs` to save results

**Initialization**:
- Unified `initialization` block replaces separate `init` and `restart`
- Method clearly specifies: `hartree_fock`, `random`, `from_file`, `from_checkpoint`
- Context-appropriate fields appear based on method

### Workflow Detection

The system automatically detects workflow based on configuration:

```python
def get_workflow_type(cfg: QMCConfig) -> str:
    if cfg.optimizer is None:
        return "energy_calculation"
    else:
        return "optimization"
```

### Element-Specific Basis Sets

The ansatz configuration supports two patterns:

```yaml
# Single basis for all atoms
ansatz:
  basis_name: cc-pvdz

# Element-specific basis sets
ansatz:
  basis:
    H: sto-3g
    O: 6-31g
```

### Design Principles

1. **Intuitive**: Configuration structure mirrors computational intent
2. **DRY**: No redundant fields (e.g., `continue_optimization` is implicit)
3. **Contextual**: Components appear where they make semantic sense
4. **Type-safe**: Backed by dataclasses with validation
5. **Flexible**: Easy to extend with new methods or components
## Important Design Patterns & Conventions

### Periodic Boundary Conditions
**CRITICAL**: This JAX-QMC code **always assumes periodic boundary conditions**. Molecules are treated as floating in the middle of large supercells, not as isolated systems in vacuum. This affects:
- All sampling algorithms work with periodic coordinates
- Ewald summation is used for long-range electrostatic interactions
- Basis set evaluation must handle periodic electron positions
- Energy calculations include supercell effects

### Atom Placement in Periodic Boxes - CRITICAL
**ALWAYS place atoms at the CENTER of the periodic box, not at corners.**

```yaml
# WRONG - atoms at corner (0,0,0)
atoms:
  - element: H
    pos: [0, 0, 0]
  - element: H
    pos: [1.4, 0, 0]

# CORRECT - atoms centered in 80 Bohr box
atoms:
  - element: H
    pos: [39.3, 40, 40]
  - element: H
    pos: [40.7, 40, 40]
```

**Why this matters:**
- Electron initialization uses Gaussian offsets around nuclear positions
- If atoms are at corner (0,0,0), negative offsets push electrons outside the [0, L] box
- The slice sampler's `box_bounds` assumes electrons are INSIDE the box
- Electrons outside the box cause `box_bounds` to compute invalid brackets
- This results in proposals going to positions like (y=-100, z=158) where log_pdf=-inf
- **Symptom**: Sampler returns 0 samples almost instantly

### MCMC Spin Format Conversion
The MCMC sampler and SlaterWF use different spin representations:

```python
# MCMC ElectronicState: 2D one-hot format (nelec, 2)
# spin-up = [1, 0], spin-down = [0, 1]
spins_2d = jnp.array([[1, 0], [0, 1]])  # up, down

# SlaterWF: discrete format (nelec,)
# spin-up = 0, spin-down = 1
spins_discrete = jnp.array([0, 1])  # up, down

# Conversion functions:
def spins_2d_to_discrete(spins_2d):
    """Convert 2D spin representation to discrete."""
    return jnp.argmax(spins_2d, axis=-1)

def spins_discrete_to_2d(spins_discrete, nelec):
    """Convert discrete spins to 2D one-hot."""
    spins_2d = jnp.zeros((nelec, 2))
    return spins_2d.at[jnp.arange(nelec), spins_discrete].set(1.0)
```

### Slice Sampler Debugging
When slice sampling produces 0 samples:
1. **Check initial positions** - Are any electrons outside [0, L] in any dimension?
2. **Check initial log_pdf** - Should be finite (not -inf or NaN)
3. **Bracket shrinking is exponential** - Takes ~log2(bracket_width) iterations, NOT thousands
4. If 0 samples in <1 second, electrons are likely outside the box (not slow convergence)

### Hydra Configuration Pattern
**ALWAYS use `@hydra.main` decorator pattern** for configuration-driven scripts:

```python
import hydra
from omegaconf import DictConfig
from jaxqmc.config import register_configs

# Register configs (required!)
register_configs()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Work with cfg.system, cfg.finite, cfg.ansatz
    pass
```

**NEVER manually parse YAML files** with `initialize()` and `compose()`. The project has established patterns for Hydra integration.

### Configuration Field Names
When working with atom configurations, be aware of field name variations:
- **Modern configs**: Use `element` field (e.g., `element: H`)
- **Legacy configs**: May use `wyckoff` field (e.g., `wyckoff: H`)
- **Code should handle both**: Use `getattr(atom_config, 'element', getattr(atom_config, 'wyckoff', None))`

The `UnitsConfig` uses:
- `positions` (not `position`) - for atomic coordinate units
- `lattice` - for lattice parameter units

### PySCF Integration Architecture
When integrating external quantum chemistry codes:
- **Separate concerns**: External code in dedicated modules (e.g., `jaxqmc/hartree_fock/pyscf.py`)
- **Structured results**: Use dataclasses for external code outputs, not raw objects
- **JAX conversion**: Convert all arrays to JAX format immediately
- **PySCF PBC module**: Use `from pyscf.pbc import gto, scf` (not `from pyscf import pbc`)

### Molecule Class Pattern
The `Crystal` class uses a fluent API:
```python
crystal = Crystal.cubic_lattice(30.0).add_atom(pt.H, (0,0,0)).add_atom(pt.H, (1.4,0,0))
```
- Returns `self` from `add_atom()` for method chaining
- Provides `nelec`, `nuclear_charges`, `positions` properties
- Uses Ewald summation methods: `potential_energy()`, `ion_electron_potential()`, `electron_electron_potential()`, and `ion_ion_potential` property

### Example Naming Conventions
- **General-purpose demos**: Use descriptive names reflecting the main concept (e.g., `hf_demo.py` for Hartree-Fock initialization)
- **Avoid misleading names**: Don't use `h2_*.py` for scripts that work with any molecule
- **Clear documentation**: Include usage examples in docstrings showing command-line arguments

### Error-Prone Areas to Watch
1. **Config field names**: Always check actual YAML structure before assuming field names
2. **PySCF imports**: Use correct submodule paths (`pyscf.pbc.gto.Cell`, not `pyscf.pbc.Cell`)
3. **Periodic vs molecular thinking**: Remember this code treats everything as periodic supercells
4. **Type imports**: Add `from typing import Tuple` when using type annotations
5. **Missing imports**: External modules may need explicit imports in `__init__.py` files
6. **Atom placement**: ALWAYS center atoms in the periodic box - corner placement breaks sampling
7. **Spin format mismatch**: MCMC uses 2D (nelec, 2), SlaterWF uses discrete (nelec,) - convert between them
8. **Box coordinate system**: Periodic box is [0, L], not [-L/2, L/2] - negative positions are OUTSIDE the box

## PySCF Integration - CRITICAL INFORMATION

### Unit System Conventions
**THIS IS THE MOST IMPORTANT SECTION FOR PYSCF USAGE**

PySCF has two separate unit conventions that MUST be understood:
1. **Atom positions in `mol.atom`**: Default to **Angstrom** unless `mol.unit = 'B'` is specified
2. **`eval_ao()` coordinates**: **ALWAYS** expects **Bohr**, regardless of `mol.unit`

**ALWAYS use Bohr units for consistency:**
```python
# CORRECT - Consistent Bohr units
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 1.4'
mol.unit = 'B'  # CRITICAL! Without this, positions are in Angstrom
mol.basis = 'sto-3g'
mol.build()

# eval_ao always expects Bohr
values = mol.eval_ao('GTOval_cart', coords_in_bohr)
```

### MO Coefficient Convention
```python
# PySCF MO coefficient shape and usage
mo_coeff.shape = (nbasis, nmo)  # Columns are MOs

# Transform AO to MO - NO TRANSPOSE!
mo_values = ao_values @ mo_coeff  # Correct
# NOT: mo_values = ao_values @ mo_coeff.T  # Wrong!
```

### Common PySCF Pitfalls and Solutions

1. **Translational Invariance Testing**
   - PySCF DOES have translational invariance when used correctly
   - Always use `mol.unit = 'B'` for both molecules when comparing
   - Use `mol.set_geom_()` (note underscore) to update geometry

2. **Spin Configuration**
   ```python
   # For odd-electron systems
   mol.spin = 1  # For H atom (1 unpaired electron)
   ```

3. **Basis Function Ordering**
   - PySCF uses Cartesian order: px, py, pz for p orbitals
   - d orbitals: xx, xy, xz, yy, yz, zz
   - Ordering may differ from other codes

### Testing Best Practices

1. **Test Directory Structure**
   ```
   jaxqmc/tests/wavefunctions/
   ├── test_*.py              # Actual pytest tests
   └── investigations/        # Debugging scripts (not tests)
       ├── README.md         # Explains investigation scripts
       └── test_*.py         # Investigation/debugging scripts
   ```

2. **When Comparing with PySCF**
   - ALWAYS specify `mol.unit = 'B'`
   - Verify atom coordinates with `mol.atom_coords()` (returns Bohr)
   - Use small molecules (H, H2, He) for initial validation
   - Check normalization with Monte Carlo integration

3. **Debugging Discrepancies**
   - First check units (99% of issues)
   - Then check MO coefficient convention (no transpose!)
   - Finally check basis function ordering

### Wavefunction Architecture

**Current Implementation (as of last refactoring):**
- `Wavefunction` base class: Abstract base using Equinox
- `BasisEvaluator`: Abstract interface for basis functions
- `GTOBasis`: Concrete GTO implementation with raw parameters
- `BSEMolecularBasisSet`: Basis set from BSE database (recommended for production)
  - Supports single basis for all atoms: `BSEMolecularBasisSet("sto-3g", molecule)`
  - Supports element-specific basis: `BSEMolecularBasisSet({"H": "sto-3g", "O": "6-31g"}, molecule)`
- `SlaterWF`: Slater determinant wavefunction

**Key Design Decisions:**
- Use `@cached_property` for JIT compilation
- Maintain PyTree compatibility with Equinox
- Separate spin-up and spin-down determinants in SlaterWF
- Fixed-size matrices for vmappability in SlaterWF
- P-orbital ordering: px, py, pz (matches PySCF convention)
- GTO normalization always applied: N = (2α/π)^(3/4) for s-orbitals

**Test Status**: ✅ All 70 wavefunction tests passing (100%)

### Common Iteration Patterns

1. **Creating a new test comparing with PySCF:**
   ```python
   # PySCF setup
   mol_pyscf = gto.Mole()
   mol_pyscf.atom = 'H 0 0 0'
   mol_pyscf.unit = 'B'  # NEVER FORGET THIS
   mol_pyscf.basis = 'sto-3g'
   mol_pyscf.spin = 1  # For odd electrons
   mol_pyscf.build()
   
   # Our setup - single basis for all atoms
   mol_ours = Molecule().add_atom(pt.H, (0.0, 0.0, 0.0))
   basis_ours = BSEMolecularBasisSet('sto-3g', mol_ours)
   
   # Or element-specific basis
   basis_ours = BSEMolecularBasisSet({'H': 'sto-3g'}, mol_ours)
   ```

2. **Running HF and getting MO coefficients:**
   ```python
   mf = scf.RHF(mol_pyscf)
   mf.kernel()
   mo_coeff = mf.mo_coeff  # Shape: (nbasis, nmo)
   # Use directly, DO NOT transpose
   ```

3. **When tests fail with mysterious ratios:**
   - Check units first
   - Print `mol.atom_coords()` to verify positions in Bohr
   - Compare at multiple points to identify systematic errors

## MCMC Module Architecture

### ElectronicSampler
The main sampler class that orchestrates MCMC sampling using a finite state machine (FSM) approach.

```python
from jaxqmc.mcmc import ElectronicSampler
from jaxqmc.mcmc.position import BoxSingleSliceSampler  # or other samplers

sampler = ElectronicSampler(
    log_pdf,              # Function: ElectronicState -> float
    nelec,                # Number of electrons
    lattice,              # (3, 3) lattice vectors
    supercell,            # (3,) supercell dimensions in standard coords (usually ones(3))
    BoxSingleSliceSampler(),  # Subsampler type
    dtype=jnp.float64,
)

# Run sampling
samples, chain_ids = sampler(rng, init_states, num_samples, params=None)
```

### Available Subsamplers
- `RandomDirectionSliceSampler` - Slice sampling in random directions (all electrons move together)
- `BoxSliceSampler` - Box-constrained slice sampling
- `BoxSingleSliceSampler` - Single-electron box slice sampling (one electron at a time)
- `RandomWalkMH` - Metropolis-Hastings with Gaussian proposals

### log_pdf Function Signature
```python
def log_pdf(elec_state: ElectronicState, params) -> float:
    """Return log probability density = 2 * log|psi|."""
    # elec_state.positions: (nelec, 3)
    # elec_state.spins: (nelec, 2) - 2D one-hot format!
    spins_discrete = jnp.argmax(elec_state.spins, axis=-1)  # Convert to discrete
    log_psi, _ = wavefunction(elec_state.positions, spins_discrete)
    return 2.0 * log_psi
```

### Electron Initialization
**Always initialize electrons INSIDE the periodic box:**
```python
# Initialize near nuclei (recommended)
nuclear_pos = crystal.positions  # (natoms, 3)
elec_assignments = jnp.arange(nelec) % len(nuclear_pos)
offsets = jrnd.normal(rng, shape=(num_chains, nelec, 3)) * 0.5
positions = nuclear_pos[elec_assignments][jnp.newaxis, :, :] + offsets

# Verify all positions are in [0, L]:
assert jnp.all(positions >= 0) and jnp.all(positions < lattice_size)
```

- **IMPORTANT**: Bare python commands don't work.  Use poetry.