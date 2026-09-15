"""No experiment script may read a name it never defines or imports.

Why this exists. The scripts under experiments/ are entry points: nothing imports
them, so nothing type-checks or exercises them until someone runs one on a GPU. A
`--help` smoke test reaches argparse and stops there, which is exactly how three
NameErrors survived the 2026-08-13 reorganization — an import that was never added, a
variable deleted with the call that produced it, and a function narrowed out of an
import list while a caller still used it. All three sat inside `main()`.

This is a scope check, not a linter: for every scope in the file it asks whether each
name that is read is bound somewhere visible — as an assignment, an import, a
parameter, a def/class, or a builtin. It catches the class of mistake above without
running anything.
"""

from __future__ import annotations

import ast
import builtins
import symtable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "experiments").rglob("*.py"))
PACKAGES = sorted(p for p in (REPO / "src").rglob("*.py"))

BUILTINS = set(dir(builtins))
# names the interpreter injects into a scope rather than binding in it
IMPLICIT = {
    "__file__",
    "__name__",
    "__doc__",
    "__class__",
    "__module__",
    "__qualname__",
    "__spec__",
    "__package__",
    "__builtins__",
}


def _bound(table: symtable.SymbolTable) -> set[str]:
    return {
        s.get_name()
        for s in table.get_symbols()
        if s.is_assigned() or s.is_imported() or s.is_parameter()
    }


def _free_names(table: symtable.SymbolTable, visible: set[str]) -> set[str]:
    here = visible | _bound(table)
    out = {
        s.get_name()
        for s in table.get_symbols()
        if s.is_referenced()
        and not (s.is_assigned() or s.is_imported() or s.is_parameter())
        and s.get_name() not in here
        and s.get_name() not in BUILTINS
        and s.get_name() not in IMPLICIT
    }
    for child in table.get_children():
        out |= _free_names(child, here)
    return out


@pytest.mark.parametrize(
    "path", SCRIPTS + PACKAGES, ids=lambda p: str(p.relative_to(REPO))
)
def test_every_name_is_bound(path: Path):
    table = symtable.symtable(path.read_text(), str(path), "exec")
    undefined = _free_names(table, set())
    assert not undefined, (
        f"{path.relative_to(REPO)} reads names it never binds: {sorted(undefined)}"
    )


def test_the_check_would_catch_a_missing_import():
    """A guard on the guard: the check must fail on code that is actually broken."""
    broken = "def main():\n    x = get_adapter('anticipatory')\n    return x\n"
    table = symtable.symtable(broken, "<broken>", "exec")
    assert _free_names(table, set()) == {"get_adapter"}


# --------------------------------------------------------------------------- del
# A second failure mode the scope check above cannot see. Several scripts free a
# large model with `del model` to reclaim GPU memory and keep running; reading an
# attribute of it after that point raises UnboundLocalError only when the line is
# reached, which on these scripts means after minutes of GPU work. That is how the
# `arch` record in public_probe.py briefly came to be built from a freed model.


def _names_read_after_del(func: "ast.FunctionDef") -> set[str]:
    import ast

    deleted: set[str] = set()
    bad: set[str] = set()

    def visit(node):
        # source order matters, so walk statements in order rather than ast.walk
        for stmt in ast.iter_child_nodes(node):
            if isinstance(stmt, ast.Delete):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        deleted.add(t.id)
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in ast.walk(stmt):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        deleted.discard(t.id)
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Name)
                    and isinstance(sub.ctx, ast.Load)
                    and sub.id in deleted
                ):
                    bad.add(sub.id)
            visit(stmt)

    visit(func)
    return bad


@pytest.mark.parametrize(
    "path", SCRIPTS + PACKAGES, ids=lambda p: str(p.relative_to(REPO))
)
def test_nothing_is_read_after_del(path: Path):
    import ast

    tree = ast.parse(path.read_text(), str(path))
    offenders = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = _names_read_after_del(node)
            if names:
                offenders[node.name] = sorted(names)
    assert not offenders, f"{path.relative_to(REPO)} reads freed names: {offenders}"


def test_the_del_check_would_catch_the_regression():
    """The exact shape of the bug this check was written for."""
    import ast

    broken = (
        "def main():\n"
        "    model = load()\n"
        "    del model\n"
        "    return {'arch': model.config.n_layer}\n"
    )
    fn = ast.parse(broken).body[0]
    assert _names_read_after_del(fn) == {"model"}
