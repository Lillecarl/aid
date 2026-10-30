# Reverse-engineering investigations

An investigation is a question about hardware or firmware that sessions answer from evidence. The first one asks
how to monitor and control the fan speed of a machine whose fans the operating system does not expose. The plugin
is `plugins/reverse/`. It gives sessions analysis tools and a record of what they find, and the operator manages
the investigation like an audit campaign (docs/audit-campaigns.md).

## Parts

- **Workspace.** One directory per investigation, a repository the operator chose. Sessions read, write and run
  programs there. Their shell and `python` run under bwrap with no network (`NativeSpec.sandbox`), so the
  workspace is the only place they change.
- **Tools on the path.** The `reverse-tools` Nix environment (`AID_REVERSE_TOOLS`): Ghidra with pyghidra, binwalk,
  unblob, acpica-tools (`acpixtract`, `iasl`), UEFITool's `uefiextract` and rizin. Sessions run them from the
  shell, inside the sandbox.
- **Ghidra tools.** Structured tools over one headless Ghidra per session (`ghidra_*`): import and analyse a file,
  list functions, decompile, disassemble, cross-references, strings, memory blocks. A helper process keeps the
  project open between calls; the project lives in the workspace under `ghidra/`, so a restart keeps the
  analysis. A raw image (an embedded controller's code, a flash dump) takes a Ghidra language id and a base
  address; analysis starts from an `entry` function at its first byte. Analysing a 2,507-function ELF took 42 s,
  starting the JVM 2 s (measured 2026-10-30, Ghidra 12.1.4, pyghidra 3.1.0).
- **Machine probes.** The machines under study are reached over SSH, from the worker and never from the sandbox.
  Read probes are fixed commands (identity, `dmidecode`, `acpidump`, hwmon, thermal zones, `fwupdmgr`, reading a
  file under `/sys` or `/proc`): they run without asking and save their output into the workspace.
  `remote_command` runs anything else and asks a person every time (`tool_permissions`), because a wrong write to
  an embedded controller or a fan can overheat or damage the machine.
- **Evidence.** `record_evidence` writes one row of `reverse.db`: an observation (seen in a tool's output or the
  bytes), an inference (reasoned from observations it names), a hypothesis (to test) or a question (unknown).
  Each row names its artifact by hash, its location (address, offset, ACPI path) and the command that showed it.
  `settle_evidence` marks a hypothesis or question confirmed, refuted or answered. A person reviews the record
  with `aid investigation query`.
- **Sessions.** `aid investigation start CONFIG.json` makes `<name>.lead` on the chosen model, with the tools
  above, and a supervisor `<name>` that watches it. The goal is the lead's first prompt.

## Order of work for a fan question

1. Probe the machine: identity, hwmon, thermal zones, firmware versions, `acpidump`.
2. Disassemble the ACPI tables (`acpixtract`, `iasl -d`). Fan methods (`_FST`, `_FSL`, `_FIF`), thermal zones and
   embedded-controller `OperationRegion` fields can answer the question with no firmware decompilation.
3. Only if the tables leave the fan to firmware: get the firmware (a capsule from the vendor or LVFS, a flash
   dump), extract its regions, and find the controller's code in Ghidra.
4. Test a hypothesis on the machine with `remote_command`, which a person approves each time.

## Not covered

- Native sessions take no MCP servers (Lillecarl/aid#55), so REA's MCP tools are not used.
- The sandbox does not stop a program in the workspace from reading the rest of the filesystem (Lillecarl/aid#44).
