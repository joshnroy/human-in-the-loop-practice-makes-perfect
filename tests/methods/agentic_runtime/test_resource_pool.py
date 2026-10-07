"""All inference and policy containers share a two-slot bounded resource pool."""

import threading
from types import SimpleNamespace

from hitl_pmp.agentic_runtime.sandbox import DockerContainer, SandboxSettings


def test_third_container_waits_until_cleanup(*, tmp_path, monkeypatch):
    settings = SandboxSettings(
        image="test", robocode_checkout=tmp_path, resource_pool_dir=tmp_path / "pool"
    )
    containers = [DockerContainer(settings=settings) for _ in range(3)]
    monkeypatch.setattr(DockerContainer, "client", staticmethod(lambda: ["docker"]))
    monkeypatch.setattr(DockerContainer, "image_id", lambda _: "sha256:" + "a" * 64)
    monkeypatch.setattr(
        "hitl_pmp.agentic_runtime.sandbox.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="false", stderr=""),
    )
    first = containers[0].command(name="first", work_dir=tmp_path, argv=["true"])
    containers[1].command(name="second", work_dir=tmp_path, argv=["true"])
    assert first[first.index("--cpus") + 1] == "3"
    assert first[first.index("--memory") + 1] == "7680m"
    admitted = threading.Event()

    def third():
        containers[2].command(name="third", work_dir=tmp_path, argv=["true"])
        admitted.set()

    worker = threading.Thread(target=third, daemon=True)
    worker.start()
    assert not admitted.wait(0.1)
    containers[0].remove(name="first")
    assert admitted.wait(2)
    containers[1].remove(name="second")
    containers[2].remove(name="third")
    worker.join(2)


def test_removed_container_slot_accepts_lowercase_docker_error(*, tmp_path, monkeypatch):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "docker-slot-0.lock").write_text("removed-container")
    settings = SandboxSettings(image="test", robocode_checkout=tmp_path, resource_pool_dir=pool)
    container = DockerContainer(settings=settings)
    monkeypatch.setattr(DockerContainer, "client", staticmethod(lambda: ["docker"]))
    monkeypatch.setattr(
        "hitl_pmp.agentic_runtime.sandbox.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="error: no such object: removed-container"
        ),
    )
    container.acquire_slot(name="replacement")
    assert (pool / "docker-slot-0.lock").read_text() == "replacement"
    container._lease.close()
