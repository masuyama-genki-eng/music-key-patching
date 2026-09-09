"""generate_masked decides by ARITY whether an editor wants the batch's prompt
indices, so an editor that takes a bound target as a second parameter silently
receives that list instead. This pins the contract for every driver that builds a
per-target editor: the callable handed to generate_masked takes one parameter.
Introduced after confirmatory_test.py was found broken by the 2026-08-22 change.
"""
import ast
import inspect
from pathlib import Path

from src.intervene.token_masks import generate_masked

REPO = Path(__file__).resolve().parents[1]
DRIVERS = ["experiments/confirmatory/confirmatory_test.py",
           "experiments/confirmatory/dump_continuations.py"]


def test_generate_masked_still_switches_on_arity():
    src = inspect.getsource(generate_masked)
    assert "wants_group" in src and "editor_fn(plen, group)" in src, (
        "the arity switch this test protects against has moved or gone; "
        "re-derive what the drivers must satisfy before deleting this test")


def test_no_driver_passes_a_two_parameter_editor():
    for rel in DRIVERS:
        tree = ast.parse((REPO / rel).read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "gen"):
                continue
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.Lambda):
                n = len(arg.args.args) + len(arg.args.posonlyargs)
                assert n == 1, (
                    f"{rel}:{node.lineno} hands generate_masked a {n}-parameter "
                    "lambda; it will receive the batch's prompt indices as its "
                    "second argument, not the bound target key")
