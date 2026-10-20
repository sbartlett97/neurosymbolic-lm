"""Deterministic code annotation via AST parsing.

GLiNER2 is trained on natural language and is unreliable on code spans, so
code segments are annotated with a parser instead: exact, never-hallucinated
name spans for functions, classes, imports, calls, and raised exceptions,
emitted in the same taxonomy vocabulary as the prose annotators.

Python is handled with the stdlib ``ast`` module (exact positions, zero
dependencies). Other languages can be plugged in by registering a parser in
``CodeAnnotator.PARSERS`` (e.g. a tree-sitter backed one); unknown languages
return an empty annotation rather than guessing.
"""

import ast
from typing import Any, Dict, List, Optional, Tuple


def _line_starts(source: str) -> List[int]:
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)
    return starts


class _PyVisitor(ast.NodeVisitor):
    """Collects taxonomy-aligned entities and relations from a Python AST."""

    def __init__(self, source: str, max_entities: int):
        self.source = source
        self.max_entities = max_entities
        self.starts = _line_starts(source)

        self.entities: List[str] = []
        self.spans: List[List[int]] = []
        self.concepts: List[List[str]] = []
        self.entity_types: List[str] = []
        self.relations: List[List[Any]] = []

        self._by_key: Dict[Tuple[str, str], int] = {}  # (name, concept) -> idx
        self._rel_seen = set()
        self._scope: List[int] = []  # entity indices of enclosing defs

    # -- position helpers ---------------------------------------------------

    def _offset(self, lineno: int, col: int) -> int:
        if 1 <= lineno <= len(self.starts):
            return self.starts[lineno - 1] + col
        return 0

    def _name_span(self, node: ast.AST, name: str) -> Optional[List[int]]:
        """Char span of `name` searched from the node's start position."""
        start = self._offset(node.lineno, node.col_offset)
        end_limit = (
            self._offset(node.end_lineno, node.end_col_offset)
            if getattr(node, "end_lineno", None)
            else len(self.source)
        )
        found = self.source.find(name, start, end_limit)
        if found == -1:
            return None
        return [found, found + len(name)]

    # -- entity/relation registration ----------------------------------------

    def _add_entity(
        self, name: str, span: Optional[List[int]], concept: str, coarse: str
    ) -> Optional[int]:
        key = (name, concept)
        if key in self._by_key:
            return self._by_key[key]
        if span is None or len(self.entities) >= self.max_entities:
            return None
        idx = len(self.entities)
        self.entities.append(name)
        self.spans.append(span)
        self.concepts.append([concept])
        self.entity_types.append(coarse)
        self._by_key[key] = idx
        return idx

    def _add_relation(self, head: Optional[int], tail: Optional[int], rel: str):
        if head is None or tail is None or head == tail:
            return
        key = (head, tail, rel)
        if key not in self._rel_seen:
            self._rel_seen.add(key)
            self.relations.append([head, tail, rel])

    # -- visitors -------------------------------------------------------------

    def _visit_def(self, node, concept: str):
        span = self._name_span(node, node.name)
        idx = self._add_entity(node.name, span, concept, "code_construct")
        if idx is not None and self._scope:
            self._add_relation(idx, self._scope[-1], "defined_in")
        if idx is not None:
            self._scope.append(idx)
            self.generic_visit(node)
            self._scope.pop()
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._visit_def(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self._visit_def(node, "function")

    def visit_ClassDef(self, node):
        self._visit_def(node, "class_name")

    def visit_Import(self, node):
        for alias in node.names:
            span = self._name_span(alias, alias.name)
            self._add_entity(alias.name, span, "module_or_package", "code_construct")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            span = self._name_span(node, node.module)
            self._add_entity(node.module, span, "module_or_package", "code_construct")
        self.generic_visit(node)

    def visit_Call(self, node):
        callee = None
        span = None
        if isinstance(node.func, ast.Name):
            callee = node.func.id
            span = self._name_span(node.func, callee)
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
            # attr sits at the end of the Attribute node
            end = self._offset(node.func.end_lineno, node.func.end_col_offset)
            span = [end - len(callee), end]
            if self.source[span[0]:span[1]] != callee:
                span = None
        if callee and self._scope:
            idx = self._add_entity(callee, span, "function", "code_construct")
            self._add_relation(self._scope[-1], idx, "calls")
        self.generic_visit(node)

    def visit_Raise(self, node):
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Name):
            span = self._name_span(exc, exc.id)
            idx = self._add_entity(exc.id, span, "exception_type", "code_construct")
            if self._scope:
                self._add_relation(self._scope[-1], idx, "raises")
        self.generic_visit(node)


def _annotate_python(code: str, max_entities: int) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _empty()
    visitor = _PyVisitor(code, max_entities)
    visitor.visit(tree)
    return {
        "entities": visitor.entities,
        "entity_spans": visitor.spans,
        "concepts": visitor.concepts,
        "entity_types": visitor.entity_types,
        "relations": visitor.relations,
    }


def _empty() -> Dict[str, Any]:
    return {
        "entities": [],
        "entity_spans": [],
        "concepts": [],
        "entity_types": [],
        "relations": [],
    }


class CodeAnnotator:
    """Annotate source code with taxonomy-aligned symbolic structure.

    Spans are character offsets into the given code string (end-exclusive);
    callers embedding code in a larger text are responsible for shifting.
    """

    PARSERS = {
        "python": _annotate_python,
        "py": _annotate_python,
        "python3": _annotate_python,
    }

    def __init__(self, max_entities: int = 16):
        self.max_entities = max_entities

    def supports(self, language: Optional[str]) -> bool:
        return bool(language) and language.lower() in self.PARSERS

    def annotate(self, code: str, language: Optional[str] = "python") -> Dict[str, Any]:
        """Return {entities, entity_spans, concepts, entity_types, relations}."""
        if not code.strip() or not self.supports(language):
            return _empty()
        result = self.PARSERS[language.lower()](code, self.max_entities)
        # Guarantee span integrity: drop anything that doesn't point at its
        # entity (defensive; the visitors only emit verified spans).
        keep = [
            i
            for i, (ent, span) in enumerate(zip(result["entities"], result["entity_spans"]))
            if code[span[0]:span[1]] == ent
        ]
        if len(keep) != len(result["entities"]):
            remap = {old: new for new, old in enumerate(keep)}
            result = {
                "entities": [result["entities"][i] for i in keep],
                "entity_spans": [result["entity_spans"][i] for i in keep],
                "concepts": [result["concepts"][i] for i in keep],
                "entity_types": [result["entity_types"][i] for i in keep],
                "relations": [
                    [remap[h], remap[t], r]
                    for h, t, r in result["relations"]
                    if h in remap and t in remap
                ],
            }
        return result
