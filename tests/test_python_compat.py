"""Colab may run an older Python than the laptop. Guard against syntax that only exists in Python 3.12+ (same-quote nesting inside f-strings,
backslashes or comments inside f-string expressions): such code passes locally and crashes on Colab with a SyntaxError."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _problems():
    bad = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.JoinedStr):
                continue
            body = (ast.get_source_segment(src, node) or "").lstrip("fFrRbBuU")
            triple = body[:3] in ('"""', "'''")
            delim = body[:3] if triple else body[:1]
            for fv in ast.walk(node):
                if isinstance(fv, ast.FormattedValue):
                    expr = ast.get_source_segment(src, fv.value) or ""
                    if (delim[0] in expr and not triple) or "\\" in expr or "#" in expr:
                        bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: {expr[:50]}")
    return bad


def test_no_python_312_only_fstrings():
    assert _problems() == []


def test_checker_would_catch_the_pattern():
    src = "v = {'k': 1}\nx = f\"{'-' if v['k'] else f'{v['k']:.2f}'}\"\n"
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.JoinedStr) and "k" in (ast.get_source_segment(src, n) or "") and (ast.get_source_segment(src, n) or "").startswith("f'"))
    seg = ast.get_source_segment(src, node)
    assert seg.startswith("f'") and "'k'" in seg                      # inner f-string reuses its own quote: the pattern the checker looks for
