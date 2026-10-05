from vivarium_runner import Machines


async def test(vms: Machines) -> None:
    await vms.aid.succeed("aid-browser-tests", timeout=1500)
