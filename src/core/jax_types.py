# src/core/jax_types.py
from __future__ import annotations
import typing as tp
from typing import Any, TypeAlias, TypeVar, Annotated
from typing import Iterable, Mapping, Hashable
from typing import TYPE_CHECKING
from typing_extensions import override

T = TypeVar(name="T")
_T = TypeVar(name="_T")
DTypeT = TypeVar("DTypeT")
ReturnValue = TypeVar(name='ReturnValue')

if TYPE_CHECKING:
  import numpy as onp
  # import jax.numpy as jnp
  from jaxtyping import (
    Array as _Array, 
    Int as Int,
    Integer as Integer,
    Float as Float, 
    Complex as Complex,
  )
  from numpy.typing import NDArray as _NP_NDArray
  NDArray: TypeAlias = _NP_NDArray[onp.generic]
  NumericNDArray: TypeAlias = _NP_NDArray[onp.number]
  IntNDArray: TypeAlias = _NP_NDArray[onp.integer]
  Array: TypeAlias = _Array
  ArrayLike: TypeAlias = Array | NumericNDArray

else:
  # runtime-only placeholder; doesn't matter for execution
  NDArray = Any
  NumericNDArray: TypeAlias = Any
  IntNDArray = Any
  Array = Any
  ArrayLike: TypeAlias = Any
  Int: TypeAlias = Annotated
  Integer: TypeAlias = Annotated
  Float: TypeAlias = Annotated
  Complex: TypeAlias = Annotated


ScalarArrayLike = Any


RNGKey = Any
r""" JAX-like alias for `numpy.random.Generator` """




Shaped: TypeAlias = object
Scalar: TypeAlias = object
PRNGKey: TypeAlias = object
DType: TypeAlias = object
Numeric: TypeAlias = object

PyTree: TypeAlias = _T | Iterable["PyTree[_T]"] | Mapping[Hashable, "PyTree[_T]"]
PyTreeDef: TypeAlias = object

@tp.runtime_checkable
class HashablePartial(tp.Protocol):
  def __call__(self, *args: object, **kwargs: object) -> object: ...
  @override
  def __hash__(self) -> int: ...

LinearMap: TypeAlias = object
Automorphism: TypeAlias = object
BatchedLinearMap: TypeAlias = object
AutomorphismBatch: TypeAlias = object


ArrayPair: TypeAlias = object
ArrayTriple: TypeAlias = object
ArrayQuad: TypeAlias = object
PyTreeX: TypeAlias = object  # domain
PyTreeY: TypeAlias = object # codomain
StateT: TypeAlias  = object
Params: TypeAlias  = object
BatchT: TypeAlias = object # (inputs, targets)


LogPDF: TypeAlias  = object
r""" Log Probability Density Function """
Distribution: TypeAlias = object
r""" Alias for distribution functions """
Kernel: TypeAlias  = object
r""" Kernel """

VJPDef = Any


@tp.runtime_checkable
class SquareMatmul(tp.Protocol):
  @property
  def shape(self) -> tuple[int, int]: ...
  @property
  def dtype(self) -> DType: ...
  def __matmul__(self, x: ArrayLike, /) -> ArrayLike: ...







# pytree class wrapper
# from jaxtyping._pytree_type import PyTree as _PyTree
# PyTree = _T|Iterable[_T]|Mapping[Hashable, _T]|_PyTree
# PyTree: TypeAlias = _PyTree
# PyTree: TypeAlias = T | Iterable["PyTree[T]"] | Mapping[Any, "PyTree[T]"]
# PyTree = _PyTree
# r""" PyTree types """
# PyTree[Array] # -> Should succeed
# PyTree.__module__ = "builtins"