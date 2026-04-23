# src/core/jax_types.pyi
from __future__ import annotations
from typing import TypeAlias, TypeVar, Protocol
from collections.abc import Callable, Hashable, Mapping, Sequence

import numpy as onp
from numpy.typing import NDArray as _NP_NDArray
from jaxtyping import (
  Shaped as Shaped, 
  Int as Int,
  Integer as Integer,
  Float as Float, 
  Complex as Complex,
)
from jax._src.basearray import Array as _Array
from jax import custom_vjp
from jax.typing import DTypeLike
import jax._src.prng as prng
from jax._src.lib import pytree
from jax._src.util import HashablePartial as _HP


from .static_types import StaticScalar

T = TypeVar(name="T")
_T = TypeVar(name="_T")
DTypeT = TypeVar(name="DTypeT", bound=onp.generic)
ReturnValue = TypeVar(name='ReturnValue')

# @typing.runtime_checkable
class SupportsDType(Protocol):
  @property
  def dtype(self) -> DType: ...

DType: TypeAlias = DTypeLike|DTypeT|onp.dtype[DTypeT]


Array: TypeAlias = _Array
Numeric: TypeAlias = Array | StaticScalar

NDArray: TypeAlias = _NP_NDArray[onp.generic]
NumericNDArray: TypeAlias = _NP_NDArray[onp.number]
IntNDArray: TypeAlias = _NP_NDArray[onp.integer]
ArrayLike: TypeAlias = Array | NumericNDArray
ScalarArrayLike: TypeAlias = Array | NumericNDArray | StaticScalar




# TODO could just replace with `Annotated` from typing, since that's what these are... or add my own thing
Scalar: TypeAlias = Shaped[Array, ""]


# PRNGKeyArray typing moves around across JAX versions; this is the pragmatic stable one.
from random import Random
PRNGKey: TypeAlias = Array|prng.PRNGKeyArray
RNGKey: TypeAlias = onp.random.Generator|Random



PyTree: TypeAlias = (
  T
  | list["PyTree[T]"]
  | Sequence["PyTree[T]"]
  | tuple["PyTree[T]", ...]
  | dict[Hashable, "PyTree[T]"]
  | Mapping[Hashable, "PyTree[T]"]

)

PyTreeDef: TypeAlias = pytree.PyTreeDef
HashablePartial: TypeAlias = _HP

LogPDF: TypeAlias  = Callable[[Array, Array], Scalar]
r""" Log Probability Density Function """
Distribution: TypeAlias = Callable[[Array, Array], Scalar]
r""" Alias for distribution functions """
Kernel: TypeAlias  = Callable[[Array, Array], Array]
r""" Kernel """

VJPDef: TypeAlias = custom_vjp[ReturnValue]|Callable[..., ReturnValue]



LinearMap: TypeAlias = Callable[[ArrayLike], ArrayLike]
Automorphism: TypeAlias = Callable[[ArrayLike], ArrayLike]
BatchedLinearMap: TypeAlias = Callable[[ArrayLike], ArrayLike]
AutomorphismBatch: TypeAlias = Callable[[ArrayLike], ArrayLike]




ArrayPair: TypeAlias = tuple[Array, Array]
ArrayTriple: TypeAlias = tuple[Array, Array, Array]
ArrayQuad: TypeAlias = tuple[Array, Array, Array, Array]
PyTreeX: TypeAlias = PyTree  # domain
PyTreeY: TypeAlias = PyTree # codomain
StateT: TypeAlias  = PyTree
Params: TypeAlias  = PyTree
BatchT: TypeAlias = tuple[Array, Array] # (inputs, targets)
