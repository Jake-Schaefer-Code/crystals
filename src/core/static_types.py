# src/core/static_types.py
from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence, Callable
from typing import Any, Protocol #, runtime_checkable, Any
# from typing_extensions import dataclass_transform # TypeAliasType
from typing_extensions import Self
import dataclasses as dcls

__all__ = [
  # morphism aliases
  "Hom",
  "ArgHom",
  "BinHom",
  "BinOp",
  "BinMap",
  "SKernel",
  "TriHom",
  "Endo",
  "EndoArgs",
  "Auto",
  # value aliases
  "Shape",
  "Series",
  "StaticScalar",
  # protocols and mixins
  "Monoid",
  "ReplaceableLike",
  "ReplaceMixin",
  "Loadable",
  "PyTreeReplaceMixin",
  "Replaceable",
  "PyTreeReplaceable",
]

# if TYPE_CHECKING:
#   # typeshed-only; not guaranteed to exist at runtime
#   from dataclasses import DataclassInstance as _DataclassInstance
# else:
#   _DataclassInstance = object  # runtime placeholder


# Callable contracts describe domains and codomains, not algebraic laws.
# BinHom/TriHom retain separate Python arguments for product domains.
type Hom[X, Y] = Callable[[X], Y]
type ArgHom[**Args, Y] = Callable[Args, Y]
type BinHom[X1, X2, Y] = Callable[[X1, X2], Y]
type BinOp[X] = Callable[[X, X], X]
type BinMap[X, Y] = Callable[[X, X], Y]
type SKernel[X, K] = Callable[[X, X], K]
type TriHom[X1, X2, X3, Y] = Callable[[X1, X2, X3], Y]
type Endo[X] = Hom[X, X]
type EndoArgs[X, Y] = Callable[[X, Y], tuple[X, Y]]
# Invertibility requires a runtime law/test; this alias cannot prove it.
type Auto[X] = Endo[X]

type Shape = Sequence[int]
r""" type for array-shape-like objects """

type Series[T] = list[tuple[int, T]]
r""" Scalar time series, positive integer steps """

StaticScalar = complex|float|int|bool



class Monoid(Protocol):
  @classmethod
  def empty(cls) -> Self: ...
  def combine(self, other: Monoid): ...


class ReplaceableLike(Protocol):
  def replace(self: Self, /, **kw: object) -> Self: ...
  def asdict(self) -> dict[str, Any]: ...
  @classmethod
  def from_dict(cls, /, **kw: object) -> Self: ...

class ReplaceMixin:
  # __slots__ = ("__weakref__",)
  def replace(self: Self, /, **kw: object) -> Self:
    return dcls.replace(self, **kw) # pyright: ignore[reportArgumentType]
  def asdict(self) -> dict[str, Any]:
    return dcls.asdict(self) # pyright: ignore[reportArgumentType]
  @classmethod
  def from_dict(cls, /, **kw: object) -> Self:
    return cls(**kw)

class Loadable(Protocol):
  @classmethod
  def load(cls, path: Path) -> Self|None: ...


class PyTreeReplaceMixin:
  # to mimic a dataclass
  # __dataclass_fields__: dict[str, object]
  def replace(self: Self, /, **kw: object) -> Self:
    import jax_dataclasses as jdc

    # Use jax_dataclasses.replace, which understands pytree_dataclass metadata
    return jdc.replace(self, **kw)  # pyright: ignore[reportArgumentType]

  def asdict(self) -> dict[str, object]:
    # jdc.asdict delegates correctly as well, but dataclasses.asdict is fine for plain cases
    return dcls.asdict(self)  # pyright: ignore[reportArgumentType]
  @classmethod
  def from_dict(cls, /, **kw: object) -> PyTreeReplaceMixin:
    return cls(**kw)

class Replaceable(Protocol):
  def replace(self: Self, /, **kwargs: object) -> Self: ...

class PyTreeReplaceable[CT](Protocol):
  def replace(self, **kw: object) -> PyTreeReplaceable[CT]: ...

  # @classmethod
  # def from_dict(cls: Type[Self], /, **kw) -> Self:
  #   return cls(**kw)

