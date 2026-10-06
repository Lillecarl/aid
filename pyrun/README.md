# pyrun

Run processes from async Python, the way pyedit edits files: Python is the language, every process is recorded,
and nothing outlives the scope that started it. Built for agents. It lives in the aid repository as a project of
its own, so it can move out.

Linux only: process groups, `/proc` and `PR_SET_CHILD_SUBREAPER`.

## Commands

A `Command` is a value: an argv list and how to run it. Awaiting one runs it in the current scope.

    from pyrun import cmd, run

    r = await run("git", "status", "--short")          # same as: await cmd("git", "status", "--short")
    r.text, r.lines, r.json(), r.code, r.stderr, r.duration, r.id

    await run("grep", "x", "notes.txt", check=False)   # a non-zero exit is a result, not an error
    await run("jq", ".a", input='{"a": 1}')
    await run("make", cwd="sub", env={"CC": "clang"}, timeout=60)

There is no shell. An argv is a list of strings, and nothing splits or expands it. `sh("...")` is the explicit
way to ask for one.

A non-zero exit raises `Failed`, which carries the `Result`: argv, exit code, and the tail of stderr.

## Pipelines, streams, parallel runs

    r = await (cmd("rg", "-n", "TODO") | cmd("head", "-5"))    # pipefail: any stage failing fails the pipeline

    async for line in cmd("pytest", "-x").lines():            # stdout as it comes; raises at the end on failure
        ...

    async with cmd("server", "--port", "0").start() as p:     # a running process: p.stdin, p.stdout, p.wait()
        port = await p.stdout.readline()

    a, b = await pyrun.all(cmd("make", "a"), cmd("make", "b"))  # together; the first failure stops the rest

## Scopes

    async with pyrun.Scope(cwd=".", env={...}, store=path, policy=ask, timeout=600):
        ...

A scope owns every process started in it. When it exits, by return, error or cancel, it kills what is left:
each process's whole group, and descendants that left their group. A timeout, the command's or the scope's,
does the same and raises `TimedOut`, which names what was still running and the last lines each process wrote.

`policy` sees each `Command` before its process starts: `async (Command) -> bool`. False raises `Denied`. aid
asks a person here.

## Records

Every process gets an id, `<scope>.<n>`, and a directory in the scope's store: `command.json` (argv, cwd, the
env it changed, start, end, exit code or signal) and `stdout`, `stderr`, written as the process writes them.
`Result.text` reads them; memory holds at most `MEMORY_LIMIT` bytes of each.

## Scripts

`python -m pyrun.script` runs an async Python script from stdin in a scope of its own: top-level `await`, with
`run`, `cmd`, `sh`, `pyrun` and `Path` in scope. Its host, `pyrun.host.run_script(source, policy=...)`, starts
it as a child process, answers its policy questions over a pipe, and returns a `Report`.

The report is what an agent reads, in this shape:

    $ git status --short                          exit 0    0.04s   k3f9.1
      M src/app.py
    $ pytest -x                                   exit 1   12.30s   k3f9.2
      … 840 lines (38 KiB) in stdout, read k3f9.2 for all of it
      <the last lines>
    printed:
      <what the script printed>
    error:
      <the traceback, when the script raised>

A process's output is clipped to its head and tail; the full output stays in the store, by id.

## Shell sessions

`ShellSession` holds one bash open across calls, so `cd` and the environment persist:

    shell = pyrun.ShellSession(cwd=".", store=path)
    first = await shell.run("cd sub && pwd")     # first.code == 0
    second = await shell.run("pwd")              # still in sub
    await shell.restart()                        # kill it now; the next run starts fresh
    await shell.aclose()                         # kill it for good

Each command is wrapped so the shell reports its own exit code on a random framing line; stdout and
stderr merge, in the order written, and stdin is empty. Commands run one at a time. A timeout kills the
shell — framing past that point is untrustworthy — and the next run starts fresh and says the state
reset. A shell the command itself ends (`exit`) respawns the same way. Every incarnation is recorded
under the session's store like any other process.
