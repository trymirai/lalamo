import jax
from jax.sharding import AbstractMesh, Mesh


def supports_mosaic_gpu(mesh: Mesh | AbstractMesh, minimum_compute_capability: int) -> bool:
    abstract_device = mesh.abstract_mesh.abstract_device
    if abstract_device is None:
        return False
    return any(
        device.device_kind == abstract_device.device_kind
        and device.device_kind.startswith("NVIDIA")
        and float(getattr(device, "compute_capability", 0)) >= minimum_compute_capability
        for device in jax.devices(abstract_device.platform)
    )
