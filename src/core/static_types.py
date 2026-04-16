# src/core/static_types.py
from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence, Callable
from typing import Any, TypeAlias, TypeVar, Generic, Protocol #, runtime_checkable, Any
# from typing_extensions import dataclass_transform # TypeAliasType
from typing_extensions import Self
import dataclasses as dcls

# if TYPE_CHECKING:
#   # typeshed-only; not guaranteed to exist at runtime
#   from dataclasses import DataclassInstance as _DataclassInstance
# else:
#   _DataclassInstance = object  # runtime placeholder


# Generic “morphism” types
T = TypeVar('T') # input batch/type
S = TypeVar('S')
K = TypeVar('K')
CT = TypeVar('CT', covariant=True)
U = TypeVar('U')      # output batch/type
P = TypeVar('P')      # params/state pytree
A = TypeVar('A')      # aux/metrics pytree
_T = TypeVar('_T')
_S = TypeVar('_S')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
Ti = TypeVar('Ti') # iterate?


Hom  = Callable[[T], U]
r""" General homomorphism (unary) """

BinHom = Callable[[T1, T2], U]
r""" Binary homomorphism (two arguments) """

BinOp = Callable[[S, S], S]
BinMap = Callable[[S, S], U]
SKernel = Callable[[S, S], K]

TriHom = Callable[[T1, T2, T3], U]
r""" Ternary homomorphism (three arguments) """

Endo = Callable[[T], T]
r""" General Endomorphism """

Auto = Endo[T] # automorphism can't encode bijectivity in typing -> synonym
r""" General Automorphism """

Shape = Sequence[int]
r""" type for array-shape-like objects """

Series: TypeAlias = list[tuple[int, _T]]
r""" Scalar time series, positive integer steps """

StaticScalar = complex|float|int|bool



class Monoid(Protocol):
  @classmethod
  def empty(cls) -> Self: ...
  def combine(self, other: 'Monoid'): ...


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
  def from_dict(cls, /, **kw: object) -> 'PyTreeReplaceMixin':
    return cls(**kw)

class Replaceable(Protocol):
  def replace(self: Self, /, **kwargs: object) -> Self: ...

class PyTreeReplaceable(Protocol, Generic[CT]):
  def replace(self, **kw: object) -> 'PyTreeReplaceable[CT]': ...

  # @classmethod
  # def from_dict(cls: Type[Self], /, **kw) -> Self:
  #   return cls(**kw)

