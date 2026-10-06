#!/usr/bin/env bash
# Compiles report.tex -> report.pdf using tectonic (self-contained LaTeX
# engine; no system texlive install required). Run from this directory
# (this folder) whenever report.tex changes, to keep report.pdf in sync.
#
# tectonic binary: ~/.local/bin/tectonic (downloaded from
# https://github.com/tectonic-typesetting/tectonic releases, musl static
# build, since this host's glibc is too old for the gnu build).
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
"$TECTONIC" report.tex
echo "Wrote report.pdf ($(du -h report.pdf | cut -f1))"
