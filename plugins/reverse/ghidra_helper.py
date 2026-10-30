"""One headless Ghidra for one session, answering JSON lines on stdin (docs/reverse-engineering.md).

Run by `aid-ghidra-python` (nix/reverse-tools.nix), a Python with pyghidra and no aid, inside the session's sandbox:
`python -I -c SOURCE PROJECT_DIR`, or by path for a manual run. Each request is `{"id", "op", "args"}`; each answer `{"id", "ok": true,
"result"}` or `{"id", "ok": false, "error"}`. A program is named after its file and hash, so the same bytes
imported again reuse the stored analysis, and a changed file is a new program. Ghidra logs to stdout: the protocol
gets the real stdout to itself, everything else goes to stderr.
"""

# ruff: noqa: PLC0415 -- Ghidra's Java packages import only after pyghidra.start(), so they import where used.
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT = "aid"
MAX_DECOMPILE_SECONDS = 120


def _program_name(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return f"{path.name}-{digest.hexdigest()[:12]}"


class Session:
    def __init__(self, project_dir: Path) -> None:
        import pyghidra

        pyghidra.start()
        self.pyghidra = pyghidra
        project_dir.mkdir(parents=True, exist_ok=True)
        self.project = pyghidra.open_project(str(project_dir), PROJECT, create=True)
        self.programs: dict[str, object] = {}
        self.consumers: dict[str, object] = {}

    def program(self, path: str):
        name = _program_name(Path(path))
        if name in self.programs:
            return self.programs[name]
        if self.project.getProjectData().getFile(f"/{name}") is None:
            raise LookupError(f"{path} is not imported: call ghidra_import on it first")
        program, consumer = self.pyghidra.consume_program(self.project, f"/{name}")
        self.programs[name], self.consumers[name] = program, consumer
        return program

    def import_(self, path: str, language: str | None, compiler: str | None, base_address: str | None) -> dict:
        target = Path(path)
        name = _program_name(target)
        if self.project.getProjectData().getFile(f"/{name}") is None:
            loader = self.pyghidra.program_loader().project(self.project).source(str(target)).name(name)
            if language is not None:
                loader = loader.loaders("BinaryLoader").language(language)
                if compiler is not None:
                    loader = loader.compiler(compiler)
                if base_address is not None:
                    # The headless argument's name, not the option's ("Base Address" is ignored silently).
                    loader = loader.addLoaderArg("-loader-baseAddr", base_address)
            with loader.load() as results:
                results.save(self.pyghidra.task_monitor())
            program = self.program(path)
            if language is not None:
                self.seed_entry(program)
            log = self.pyghidra.analyze(program)
            program.save("aid: analysed", self.pyghidra.task_monitor())
        else:
            program, log = self.program(path), ""
        return self.summary(program) | {"program": name, "analysis_log": log[-4000:]}

    def seed_entry(self, program) -> None:
        """A raw image has no header naming code: analysis starts from a function at its first byte, `entry`."""
        from ghidra.program.flatapi import FlatProgramAPI

        # Not getImageBase: a raw image's stays 0 wherever its block was loaded (measured, 12.1.4).
        base = program.getMemory().getMinAddress()
        with self.pyghidra.transaction(program, "aid: entry"):
            api = FlatProgramAPI(program)
            api.disassemble(base)
            if program.getFunctionManager().getFunctionAt(base) is None:
                api.createFunction(base, "entry")

    def summary(self, program) -> dict:
        memory = program.getMemory()
        return {
            "language": str(program.getLanguageID()),
            "compiler": str(program.getCompilerSpec().getCompilerSpecID()),
            "format": str(program.getExecutableFormat()),
            "image_base": str(program.getImageBase()),
            "lowest_address": str(memory.getMinAddress()),
            "functions": program.getFunctionManager().getFunctionCount(),
            "blocks": [
                f"{b.getName()} {b.getStart()}-{b.getEnd()} {'r' if b.isRead() else '-'}"
                f"{'w' if b.isWrite() else '-'}{'x' if b.isExecute() else '-'}"
                for b in memory.getBlocks()
            ],
        }

    def address(self, program, text: str):
        found = program.getAddressFactory().getAddress(text)
        if found is None:
            raise ValueError(f"{text!r} is no address in this program")
        return found

    def function(self, program, which: str):
        manager = program.getFunctionManager()
        named = [f for f in manager.getFunctions(True) if f.getName() == which]
        if len(named) > 1:
            places = ", ".join(str(f.getEntryPoint()) for f in named)
            raise ValueError(f"{len(named)} functions are named {which}: pick one by address ({places})")
        if named:
            return named[0]
        try:
            found = manager.getFunctionContaining(self.address(program, which))
        except Exception:
            found = None
        if found is None:
            raise LookupError(f"no function named {which} or containing that address")
        return found

    def functions(self, path: str, query: str | None, offset: int, limit: int) -> dict:
        program = self.program(path)
        wanted = query.lower() if query else None
        rows = []
        for f in program.getFunctionManager().getFunctions(True):
            if wanted is None or wanted in f.getName().lower():
                rows.append(f"{f.getEntryPoint()} {f.getName()} ({f.getBody().getNumAddresses()} bytes)")
        return {"total": len(rows), "rows": rows[offset : offset + limit]}

    def decompile(self, path: str, which: str) -> dict:
        from ghidra.app.decompiler import DecompInterface

        program = self.program(path)
        function = self.function(program, which)
        decompiler = DecompInterface()
        decompiler.openProgram(program)
        try:
            result = decompiler.decompileFunction(function, MAX_DECOMPILE_SECONDS, self.pyghidra.task_monitor())
            if not result.decompileCompleted():
                raise RuntimeError(f"decompiling {function.getName()} failed: {result.getErrorMessage()}")
            code = result.getDecompiledFunction().getC()
        finally:
            decompiler.dispose()
        return {"function": function.getName(), "entry": str(function.getEntryPoint()), "c": code}

    def disassemble(self, path: str, address: str, count: int) -> dict:
        program = self.program(path)
        rows = []
        for instruction in program.getListing().getInstructions(self.address(program, address), True):
            if len(rows) >= count:
                break
            raw = bytes(b & 0xFF for b in instruction.getBytes()).hex()
            rows.append(f"{instruction.getAddress()}  {raw:<16}  {instruction}")
        return {"rows": rows}

    def xrefs(self, path: str, address: str, limit: int) -> dict:
        program = self.program(path)
        manager = program.getFunctionManager()
        rows = []
        for ref in program.getReferenceManager().getReferencesTo(self.address(program, address)):
            if len(rows) >= limit:
                break
            source = ref.getFromAddress()
            caller = manager.getFunctionContaining(source)
            rows.append(f"{source} {ref.getReferenceType()} in {caller.getName() if caller else '?'}")
        return {"rows": rows}

    def strings(self, path: str, query: str | None, offset: int, limit: int) -> dict:
        program = self.program(path)
        wanted = query.lower() if query else None
        rows = []
        for data in program.getListing().getDefinedData(True):
            if not data.hasStringValue():
                continue
            text = str(data.getValue())
            if wanted is None or wanted in text.lower():
                rows.append(f"{data.getAddress()} {json.dumps(text)}")
        return {"total": len(rows), "rows": rows[offset : offset + limit]}

    def memory(self, path: str) -> dict:
        return self.summary(self.program(path))

    def close(self) -> None:
        for name, program in self.programs.items():
            program.release(self.consumers[name])
        self.project.close()


def serve(project_dir: Path, protocol) -> None:
    session = Session(project_dir)
    ops = {
        "import": lambda a: session.import_(a["path"], a.get("language"), a.get("compiler"), a.get("base_address")),
        "functions": lambda a: session.functions(a["path"], a.get("query"), a.get("offset", 0), a.get("limit", 200)),
        "decompile": lambda a: session.decompile(a["path"], a["function"]),
        "disassemble": lambda a: session.disassemble(a["path"], a["address"], a.get("count", 60)),
        "xrefs": lambda a: session.xrefs(a["path"], a["address"], a.get("limit", 200)),
        "strings": lambda a: session.strings(a["path"], a.get("query"), a.get("offset", 0), a.get("limit", 200)),
        "memory": lambda a: session.memory(a["path"]),
    }
    protocol.write(json.dumps({"id": None, "ok": True, "result": "ready"}) + "\n")
    protocol.flush()
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                answer = {"id": request["id"], "ok": True, "result": ops[request["op"]](request.get("args", {}))}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                answer = {"id": request["id"], "ok": False, "error": f"{type(error).__name__}: {error}"}
            protocol.write(json.dumps(answer) + "\n")
            protocol.flush()
    finally:
        session.close()


if __name__ == "__main__":
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    serve(Path(sys.argv[1]), protocol)
