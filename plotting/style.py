# plotting/style.py
r"""Matplotlib parameters for the paper figures (LaTeX text, Computer Modern, vector-safe fonts).

``PAPER_RCPARAMS`` needs a working LaTeX installation because it sets ``text.usetex``.
"""
from __future__ import annotations

import matplotlib as mpl

PAPER_RCPARAMS = {
  "font.size": 16,
  "axes.titlesize": 20,
  "axes.labelsize": 16,
  "xtick.labelsize": 9,
  "ytick.labelsize": 9,
  "figure.dpi": 300,
  "savefig.dpi": 300,
  "pdf.fonttype": 42,
  "ps.fonttype": 42,
  "mathtext.fontset": "dejavusans",
  "text.usetex": True,
  # "font.family": "Helvetica",
  "font.family": "Computer Modern Roman",
}


def use_paper_style() -> None:
  r""" Apply ``PAPER_RCPARAMS`` globally. """
  mpl.rcParams.update(PAPER_RCPARAMS)
