import anyio

from vivarium_runner import Machine, Machines

# A unit is active before its process listens: the daemon imports its agents, the web UI its
# OIDC provider, and UML is slow. Poll the socket the way tests/web_harness.py does.
TIMEOUT = 180


async def wait_for_http(machine: Machine, url: str) -> None:
    with anyio.fail_after(TIMEOUT):
        while True:
            rc, _ = await machine.execute(f"curl -sSf {url} -o /dev/null")
            if rc == 0:
                return
            await anyio.sleep(1)


async def test(vms: Machines) -> None:
    await vms.aid.wait_for_unit("dex.service")
    await vms.aid.wait_for_unit("aid.service")
    await vms.aid.wait_for_unit("aid-web.service")
    await wait_for_http(vms.aid, "http://127.0.0.1:5556/dex/.well-known/openid-configuration")
    await wait_for_http(vms.aid, "http://127.0.0.1:8080/healthz")
    await vms.aid.succeed("curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ | grep -q 303")
