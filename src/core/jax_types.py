# src/core/jax_types.py
from __future__ import annotations

import typing as tp
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

from typing_extensions import override

__all__ = [
  "SupportsDType",
  "DType",
  "XLADevice",
  "Array",
  "Numeric",
  "NDArray",
  "NumericNDArray",
  "IntNDArray",
  "ArrayLike",
  "ScalarArrayLike",
  "RealScalarLike",
  "Shaped",
  "Int",
  "Integer",
  "Float",
  "Complex",
  "Scalar",
  "PRNGKey",
  "RNGKey",
  "PyTree",
  "PyTreeDef",
  "HashablePartial",
  "LogPDF",
  "Distribution",
  "Kernel",
  "LinearMap",
  "Automorphism",
  "BatchedLinearMap",
  "AutomorphismBatch",
  "ArrayPair",
  "ArrayTriple",
  "ArrayQuad",
  "PyTreeX",
  "PyTreeY",
  "StateT",
  "Params",
  "BatchT",
  "JitWrapped",
  "VJPDef",
  "LowerableFn",
  "SquareMatmul",
]

if TYPE_CHECKING:
  from random import Random

  import jax.stages as jstages
  from jax._src import prng
  import numpy as onp
  from jax import Array as _Array
  from jax import Device as _Device
  from jax import custom_vjp
  from jax._src.pjit import JitWrapped as _JitWrapped
  from jax._src.util import HashablePartial as _HP
  from jax.tree_util import PyTreeDef as _PyTreeDef
  from jax.typing import DTypeLike
  from jaxtyping import (
    Complex as Complex,
    Float as Float,
    Int as Int,
    Integer as Integer,
    Shaped as Shaped,
  )
  from numpy.typing import NDArray as _NP_NDArray

  from src.core.static_types import StaticScalar

  type DType[DT: onp.generic] = DTypeLike | DT | onp.dtype[DT]
  XLADevice: TypeAlias = _Device  # pyright: ignore[reportInvalidTypeForm]

  Array: TypeAlias = _Array
  Numeric: TypeAlias = Array | StaticScalar
  NDArray: TypeAlias = _NP_NDArray[onp.generic]
  NumericNDArray: TypeAlias = _NP_NDArray[onp.number]
  IntNDArray: TypeAlias = _NP_NDArray[onp.integer]
  ArrayLike: TypeAlias = Array | NumericNDArray
  ScalarArrayLike: TypeAlias = Array | NumericNDArray | StaticScalar
  RealScalarLike: TypeAlias = int | float | Array | onp.ndarray

  Scalar: TypeAlias = Shaped[Array, ""]
  PRNGKey: TypeAlias = Array | prng.PRNGKeyArray
  RNGKey: TypeAlias = onp.random.Generator | Random

  type PyTree[Leaf] = (
    Leaf
    | list[PyTree[Leaf]]
    | Sequence[PyTree[Leaf]]
    | tuple[PyTree[Leaf], ...]
    | dict[Hashable, PyTree[Leaf]]
    | Mapping[Hashable, PyTree[Leaf]]
  )
  PyTreeDef: TypeAlias = _PyTreeDef  # pyright: ignore[reportInvalidTypeForm]
  HashablePartial: TypeAlias = _HP

  LogPDF: TypeAlias = Callable[[Array, Array], Scalar]
  Distribution: TypeAlias = Callable[[Array, Array], Scalar]
  Kernel: TypeAlias = Callable[[Array, Array], Array]

  LinearMap: TypeAlias = Callable[[ArrayLike], ArrayLike]
  Automorphism: TypeAlias = Callable[[ArrayLike], ArrayLike]
  BatchedLinearMap: TypeAlias = Callable[[ArrayLike], ArrayLike]
  AutomorphismBatch: TypeAlias = Callable[[ArrayLike], ArrayLike]

  ArrayPair: TypeAlias = tuple[Array, Array]
  ArrayTriple: TypeAlias = tuple[Array, Array, Array]
  ArrayQuad: TypeAlias = tuple[Array, Array, Array, Array]
  PyTreeX: TypeAlias = PyTree
  PyTreeY: TypeAlias = PyTree
  StateT: TypeAlias = PyTree
  Params: TypeAlias = PyTree
  BatchT: TypeAlias = tuple[Array, Array]

  JitWrapped: TypeAlias = _JitWrapped
  type VJPDef[R] = custom_vjp[R] | Callable[..., R]

else:
  # Lightweight runtime forms for aliases that are used only in annotations.
  NDArray = Any
  NumericNDArray: TypeAlias = Any
  IntNDArray = Any
  Array = Any
  ArrayLike: TypeAlias = Any
  Int: TypeAlias = Annotated
  Integer: TypeAlias = Annotated
  Float: TypeAlias = Annotated
  Complex: TypeAlias = Annotated

  RNGKey = Any
  XLADevice = Any
  RealScalarLike = Any
  ScalarArrayLike = Any

  Shaped: TypeAlias = object
  Scalar: TypeAlias = object
  PRNGKey: TypeAlias = object
  type DType[DT] = object
  Numeric: TypeAlias = object

  type PyTree[Leaf] = Leaf | Iterable[PyTree[Leaf]] | Mapping[Hashable, PyTree[Leaf]]
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
  PyTreeX: TypeAlias = object
  PyTreeY: TypeAlias = object
  StateT: TypeAlias = object
  Params: TypeAlias = object
  BatchT: TypeAlias = object

  LogPDF: TypeAlias = object
  Distribution: TypeAlias = object
  Kernel: TypeAlias = object

  type VJPDef[R] = Callable[..., R]
  JitWrapped = Any


class LowerableFn[**Args, R](tp.Protocol):
  """A compiled arrow with the same argument signature as its source function."""

  def __call__(self, *args: Args.args, **kwargs: Args.kwargs) -> R: ...
  def lower(self, *args: Args.args, **kwargs: Args.kwargs) -> jstages.Lowered: ...
  def trace(self, *args: Args.args, **kwargs: Args.kwargs) -> jstages.Traced: ...


@tp.runtime_checkable
class SupportsDType(tp.Protocol):
  @property
  def dtype(self) -> DType: ...


@tp.runtime_checkable
class SquareMatmul(tp.Protocol):
  @property
  def shape(self) -> tuple[int, int]: ...

  @property
  def dtype(self) -> DType: ...

  def __matmul__(self, x: ArrayLike, /) -> ArrayLike: ...
