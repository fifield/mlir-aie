"""M10c: make PythoC's ``@extern`` stubs external declarations (B-3, fail closed).

PythoC's compile-by-text path compiles an ``@extern`` stub in the kernel text as an *internal function with an empty
body* (proposal P). ``declare_externs`` rewrites the stub's definition in the raw IR into a declaration before ``opt``.

Fail closed:
* each requested name must match **exactly one** definition of the exact spelling PythoC emits today
  (``define internal void @"name"(<params>)`` + newline + ``{``); zero or several matches raise, and so does any other
  ``define`` of the name (``dso_local``, attribute groups, a brace on the same line: every other spelling is refused, not
  skipped); the body must be the stub's (no call, no store through the parameters);
* after the rewrite the IR must hold exactly one ``declare void @"name"(`` and no ``define`` of the name.
The object-level postconditions (``U exp_v64`` in each caller, pinned call-relocation counts, one global ``exp_v64`` per
linked ELF equal to the object's text, ``declare`` in each caller's ``.opt.ll``) are checked by ``m10c.g0``.
``externs(names)`` wraps ``m9b.compiler._strip_unsupported_flags`` for the duration of one compile only.
"""

import contextlib, re


class ExternError(RuntimeError):
    pass


def declare_externs(ir_text, names):
    for n in names:
        q = re.escape(n)
        any_def = re.findall(r'^define\b[^\n]*@"?' + q + r'"?\(', ir_text, flags=re.M)
        exact = list(
            re.finditer(
                r'^define internal void @"' + q + r'"\(([^\n]*)\)\n\{\n',
                ir_text,
                flags=re.M,
            )
        )
        if len(exact) != 1 or len(any_def) != 1:
            raise ExternError(
                f"@extern {n}: {len(exact)} exact stub definitions, {len(any_def)} definitions in all (exactly 1 and 1 required)"
            )
        m = exact[0]
        end = ir_text.index("\n}\n", m.end()) + 3
        body = ir_text[m.end() : end]
        stores = re.findall(
            r'^\s*store [^\n]*, [^\n]*\* (%"[^"]+")\s*$', body, flags=re.M
        )
        if (
            re.search(r"\bcall\b", body)
            or not all(
                re.fullmatch(r'%"(\w+_addr|tuple_field_\d+(\.\d+)?)"', t)
                for t in stores
            )
            or len(stores) != len(re.findall(r"^\s*store ", body, flags=re.M))
        ):
            raise ExternError(
                f"@extern {n}: the stub body is not empty (a call, or a store outside its own locals)"
            )
        ir_text = (
            ir_text[: m.start()]
            + f'declare void @"{n}"({m.group(1)})\n'
            + ir_text[end:]
        )
        if len(
            re.findall(r'^declare void @"' + q + r'"\(', ir_text, flags=re.M)
        ) != 1 or re.search(r'^define\b[^\n]*@"?' + q + r'"?\(', ir_text, flags=re.M):
            raise ExternError(
                f"@extern {n}: postcondition (one declaration, no definition) failed"
            )
    return ir_text


@contextlib.contextmanager
def externs(names):
    from . import compiler as C

    orig = C._strip_unsupported_flags
    seen = {"calls": 0}

    def rewritten(s):
        out = orig(s)
        if names and re.search(
            r'^define internal void @"(' + "|".join(map(re.escape, names)) + r')"\(',
            out,
            flags=re.M,
        ):
            seen["calls"] += 1
            return declare_externs(out, names)
        return out

    C._strip_unsupported_flags = rewritten
    try:
        yield seen
        if names and seen["calls"] != 1:
            raise ExternError(
                f'@extern {names}: the raw IR was rewritten {seen["calls"]} times (exactly 1 required)'
            )
    finally:
        C._strip_unsupported_flags = orig
