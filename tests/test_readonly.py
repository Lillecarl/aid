from __future__ import annotations

import pytest

from aid.readonly import argv_reads, command_reads, reads

READ_COMMANDS = [
    "ls -la a | head -30; du -sh x",
    "ls modules/nixos hosts/hetztop; echo ===; cat hosts/hetztop/default.nix | head -80",
    "p=/nix/store/ywm3-source; sed -n 630,660p $p/nixos/modules/a.nix; echo '### caddy'; grep -n \"email =\" $p/b.nix",
    "B=/nix/store/x-oauth2-proxy/bin/oauth2-proxy; $B --help 2>&1 | grep -E 'cookie-secret|upstream'",
    "cd /home/u/Code/repo && nix eval --file . --raw pkgs.postgresql.version 2>/dev/null",
    "jj --no-pager log -r 'develop..all()' --no-graph -T 'commit_id.short()'",
    "jj diff --stat -r develop..bf9f791",
    "jj file show -r @- src/a.py",
    "git -C repo log --oneline -5 && git status --short",
    "git branch -a",
    "find . -name '*.py' -newer x | wc -l",
    "grep -rn 'foo$' src || true",
    "systemctl --user status aid.service",
    "journalctl --user -u aid -n 50",
    "awk '{print $1}' file | sort | uniq -c",
    "nix flake metadata",
    "pynix path-info /nix/store/abc-x",
    "hostname; nproc; free -g; df -h /; uname -a",
    "mount | grep -E ' / '; command -v sudo",
    "timeout 6 jj --no-pager git remote list",
    "ip -4 addr show scope global",
    "sort -u names | uniq -c; tree -L 2 src; fd -e py src; rg -n todo --glob '*.py'",
    "git tag -l 'v*'; git grep -n foo; man 5 nix.conf; date +%s; hostname -f; hostnamectl",
    "f=src/a.py; wc -l $f",
]

UNSURE_COMMANDS = [
    "rm -rf ./*",
    "ls; rm -rf ~",
    "cat $(which sh)",
    "cat `which sh`",
    "echo hi > notes.txt",
    "echo hi >> notes.txt",
    "sed -i s/a/b/ file",
    "sed -n 1p --in-place file",
    "sed 's/a/b/w out.txt' file",
    "sort -o sorted.txt in.txt",
    "sort -uo sorted.txt in.txt",
    "uniq in.txt out.txt",
    "tree -o out.txt .",
    "fd -e py -x rm",
    "fd -X touch",
    "rg --pre ./run.sh foo",
    "file -C -m magic",
    "xxd in.bin out.hex",
    "bat --pager 'sh -c id' a.txt",
    "bat cache --clear",
    "man -P 'sh -c id' ls",
    "printf -v PAGER %s evil",
    "date -s tomorrow",
    "hostname evil",
    "hostnamectl set-hostname evil",
    "MANPAGER='sh -c id' man ls",
    "x=1 ls",
    "PAGER=evil; git log",
    "LD_PRELOAD=/tmp/x.so; ls",
    "jj log --config ui.pager=evil",
    "jj diff --tool evil",
    "nix eval --allow-unsafe-native-code-during-evaluation --expr 'builtins.exec [\"id\"]'",
    "nix --option allow-unsafe-native-code-during-evaluation true eval --expr 1",
    "awk -f prog.awk data",
    "gawk -l ./evil.so 1 data",
    "sed -f script.sed file",
    "git grep -O vim foo",
    "git branch --unset-upstream",
    "git branch --edit-description",
    "git tag v1",
    "find . -delete",
    "find . -exec rm {} +",
    "cat $UNSET",
    "git push",
    "git branch -D old",
    "git -c core.pager=sh log",
    "jj --config ui.pager=x log",
    "jj abandon @",
    "jj git push",
    "nix build --file . x",
    "systemctl restart aid",
    "journalctl --vacuum-time=1d",
    "awk '{system(\"rm x\")}' f",
    "cat ~/.ssh/id_ed25519",
    "cat .privrc",
    "grep -r token ~/.local/share/opencode/auth.json",
    "cat 'unclosed",
    "sleep 1 &",
    "cat <<EOF\nx\nEOF",
    "xargs rm < list",
    "env",
    "curl https://example.com",
    "python -c 'print(1)'",
    "diff <(ls a) <(ls b)",
    "mount /dev/sda1 /mnt",
    "timeout 5 rm -rf x",
    "ip addr add 10.0.0.1/24 dev eth0",
    "jj git push",
    "command rm x",
]


@pytest.mark.parametrize("command", READ_COMMANDS)
def test_plain_reads_pass(command: str) -> None:
    assert command_reads(command)


@pytest.mark.parametrize("command", UNSURE_COMMANDS)
def test_anything_unclear_is_not_a_read(command: str) -> None:
    assert not command_reads(command)


def test_argv_and_tool_shapes() -> None:
    assert argv_reads(["jj", "--no-pager", "diff", "--stat", "-r", "a..b"])
    assert argv_reads(["sh", "-c", "ls -la | head; du -sh ."])
    assert not argv_reads(["sh", "-c", "ls; rm x"])
    assert not argv_reads(["/nix/store/x/bin/tool", "--force"])
    assert argv_reads(["/nix/store/x/bin/tool", "--help"])
    assert reads("python", {"argv": ["jj", "log"], "cwd": "."})
    assert reads("shell", {"command": "ls -la"})
    assert reads("Read", {"file_path": "/repo/src/a.py"})
    assert not reads("Read", {"file_path": "/home/u/.ssh/id_rsa"})
    assert not reads("Write", {"file_path": "/repo/a.py", "content": "x"})
    assert not reads("shell", "ls")
    assert not reads("python", {"argv": ["ls"], "command": "ls"})
