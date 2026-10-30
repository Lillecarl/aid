"""The machines under study, over SSH from the worker, never from the sandbox, which has no network
(docs/reverse-engineering.md).

Read probes are fixed commands and run without asking. `machine_read` and `machine_list` take a path under /sys or
/proc, checked so it reaches the remote shell as one word. `remote_command` runs what the model wrote, so the
investigation's spec makes it ask a person every time. Each output is saved under `machines/<machine>/` in the
workspace, where the shell's tools read it.
"""

from __future__ import annotations

import os
import re
import time
from typing import Final

import anyio

import aid
from aid.coding import CODING, spill
from aid.protocol import AidError

ENV_SSH: Final = "AID_REVERSE_SSH"
"""The program run as `ssh`; tests point it at a fake that runs the command here."""
SSH_OPTIONS: Final = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
TIMEOUT: Final = 300.0
READABLE: Final = re.compile(r"/(sys|proc)(/[A-Za-z0-9_.:+@,-]+)+")
"""A path under /sys or /proc that a shell takes as one word: no space, quote, glob or `..`."""
HWMON: Final = r"""for h in /sys/class/hwmon/hwmon*; do
  echo "== $h name=$(cat $h/name 2>/dev/null) device=$(readlink -f $h/device 2>/dev/null)"
  for f in $h/*; do
    case ${f##*/} in uevent|device|power|subsystem|of_node|name) continue;; esac
    [ -f "$f" ] && printf '%s=%s\n' "${f##*/}" "$(cat "$f" 2>&1)"
  done
done"""
THERMAL: Final = r"""for z in /sys/class/thermal/thermal_zone*; do
  echo "== $z type=$(cat $z/type 2>/dev/null) temp=$(cat $z/temp 2>/dev/null) policy=$(cat $z/policy 2>/dev/null) mode=$(cat $z/mode 2>/dev/null)"
  for t in $z/trip_point_*_temp; do [ -f "$t" ] && echo "${t##*/}=$(cat $t) type=$(cat ${t%_temp}_type 2>/dev/null)"; done
done
for c in /sys/class/thermal/cooling_device*; do
  echo "== $c type=$(cat $c/type 2>/dev/null) cur_state=$(cat $c/cur_state 2>/dev/null) max_state=$(cat $c/max_state 2>/dev/null)"
done"""
IDENTITY: Final = r"""uname -a; cat /etc/os-release 2>/dev/null
for f in sys_vendor product_name product_version board_vendor board_name bios_vendor bios_version bios_date; do
  printf '%s=%s\n' $f "$(cat /sys/class/dmi/id/$f 2>/dev/null)"
done
printf 'device-tree model=%s\n' "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
grep -m 8 -E '^(model name|Hardware|CPU part|CPU implementer)' /proc/cpuinfo"""
DEVICES: Final = r"""lspci -nn 2>&1; echo; lsusb 2>&1; echo
for d in /sys/bus/acpi/devices/*; do
  printf '%s hid=%s path=%s\n' "${d##*/}" "$(cat $d/hid 2>/dev/null)" "$(cat $d/path 2>/dev/null)"
done
echo; ls /sys/bus/platform/devices; echo; ls /sys/bus/i2c/devices 2>/dev/null"""
PROBES: Final = {
    "identity": IDENTITY,
    "dmidecode": "sudo -n dmidecode 2>&1",
    "acpidump": "sudo -n acpidump 2>&1",
    "hwmon": HWMON,
    "thermal": THERMAL,
    "sensors": "sensors -j 2>&1",
    "devices": DEVICES,
    "modules": "lsmod 2>&1; echo; sudo -n dmesg 2>&1 | tail -n 3000",
    "firmware": "fwupdmgr get-devices --json 2>&1",
    "nvidia": "nvidia-smi -q 2>&1",
    "ipmi": "sudo -n ipmitool sdr elist 2>&1; echo; sudo -n ipmitool fru 2>&1",
}
"""What a read probe runs on the machine: each only reads. `sudo -n` fails at once where it would ask a password."""


async def _destination(machine: str) -> str:
    async with aid.connect() as client:
        machines = await client.call_plugin("reverse", "machines")
    if not isinstance(machines, dict) or not isinstance(dest := machines.get(machine), str):
        known = ", ".join(sorted(machines)) if isinstance(machines, dict) else ""
        raise AidError("not_found", f"no machine {machine!r} in this investigation (machines: {known or 'none'})")
    return dest


async def _run(machine: str, command: str, label: str) -> str:
    """Run `command` on the machine, save what it printed under the workspace, and return it with where it went."""
    dest = await _destination(machine)
    ssh = os.environ.get(ENV_SSH, "ssh")
    try:
        with anyio.fail_after(TIMEOUT):
            done = await anyio.run_process([ssh, *SSH_OPTIONS, dest, "--", command], check=False, input=b"")
    except TimeoutError:
        raise AidError("timeout", f"{label} on {machine} ran past {TIMEOUT:.0f} s") from None
    output = done.stdout.decode(errors="replace")
    errors = done.stderr.decode(errors="replace")
    # 255 is ssh's own failure: the machine answered nothing, and there is nothing to save.
    if done.returncode == 255:
        raise AidError("unreachable", f"ssh to {machine} failed: {errors.strip()[-2000:]}")
    folder = CODING.get().cwd / "machines" / machine
    await anyio.Path(folder).mkdir(parents=True, exist_ok=True)
    saved = folder / f"{label}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.txt"
    await anyio.Path(saved).write_text(output)
    where = saved.relative_to(CODING.get().cwd)
    head = f"[{label} on {machine}: exit {done.returncode}, saved to {where}]\n"
    tail = f"\n[stderr]\n{errors[-4000:]}" if errors.strip() else ""
    return await spill(head + output + tail)


@aid.mcptool
async def machine_probe(machine: str, probe: str) -> str:
    """Read a machine under study with one fixed probe; nothing on it changes. `probe`: identity (kernel, OS, DMI
    and device-tree names), dmidecode, acpidump (every ACPI table, for acpixtract and iasl), hwmon (every sensor,
    fan and pwm file), thermal (zones, trip points, cooling devices), sensors, devices (PCI, USB, ACPI and platform
    devices), modules (loaded modules and the kernel log), firmware (fwupd's devices and versions), nvidia, ipmi
    (a BMC's sensors and FRU). The output is saved under machines/<machine>/ in your workspace."""
    if (command := PROBES.get(probe)) is None:
        raise AidError("bad_args", f"no probe {probe!r}: {', '.join(PROBES)}")
    return await _run(machine, command, probe)


def _readable(path: str) -> str:
    if not READABLE.fullmatch(path) or "/../" in f"{path}/" or "/./" in f"{path}/":
        raise AidError("bad_args", f"{path!r} is no plain path under /sys or /proc")
    return path


@aid.mcptool
async def machine_read(machine: str, path: str, as_hex: bool = False) -> str:
    """Read one file under /sys or /proc on a machine under study, as root where sudo allows it: a sensor, a
    register file such as /sys/kernel/debug/ec/ec0/io, an ACPI table under /sys/firmware/acpi/tables. `as_hex`:
    dump the bytes with their offsets, for binary files."""
    quoted = _readable(path)
    dump = " | od -A x -t x1z -v" if as_hex else ""
    command = f"{{ sudo -n cat -- {quoted} 2>/dev/null || cat -- {quoted}; }}{dump}"
    return await _run(machine, command, f"read{quoted.replace('/', '_')}")


@aid.mcptool
async def machine_list(machine: str, path: str) -> str:
    """List one directory under /sys or /proc on a machine under study, with links shown and their targets."""
    quoted = _readable(path)
    return await _run(machine, f"ls -la -- {quoted}", f"ls{quoted.replace('/', '_')}")


@aid.mcptool
async def remote_command(machine: str, command: str, why: str) -> str:
    """Run a shell command on a machine under study: the one way to change anything there, such as writing a
    fan or controller register to test a hypothesis. A person reads `command` and `why` and approves each call; a
    wrong write can overheat or damage the machine, so read first and write the least you can. `why`: what you test
    and what you expect to see."""
    del why  # For the person who approves the call: the permission request shows every argument.
    return await _run(machine, command, "command")
