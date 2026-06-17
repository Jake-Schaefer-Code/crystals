

from __future__ import annotations
from typing import TypeVar, Generic, Protocol


class Field(Protocol):
  def __mul__(self, other) -> 'Field': ...
  def __add__(self, other) -> 'Field': ...
  def __truediv__(self, other) -> 'Field': ...
  def conjugate(self) -> 'Field': ...
  

F = TypeVar('F', bound=Field)
