"""DeepMD se_a descriptor adapter for CACE atomistic batches."""

import torch
from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.utils import env
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list


def build_descriptor(config):
    options = dict(config)
    options.pop("type", None)
    return DescrptSeA(**options)


def compute_descriptor_stats(descriptor, systems, batch_size=1):
    if batch_size < 1:
        raise ValueError("descriptor statistics batch_size must be positive")

    samples = []
    for system in systems:
        nframes = int(system["coord"].shape[0])
        for start in range(0, nframes, batch_size):
            stop = min(start + batch_size, nframes)
            sample = {}
            for key, value in system.items():
                if torch.is_tensor(value) and key in ("coord", "atype", "box"):
                    value = value[start:stop]
                    dtype = (
                        env.GLOBAL_PT_FLOAT_PRECISION
                        if key in ("coord", "box") else value.dtype
                    )
                    value = value.to(device=env.DEVICE, dtype=dtype)
                sample[key] = value
            samples.append(sample)
    descriptor.compute_input_stats(samples)
    return descriptor


class DeepmdSeAInput(torch.nn.Module):
    """Translate a CACE graph batch into DeepMD se_a input and node features."""

    def __init__(self, descriptor, type_map, output_key="node_feats"):
        super().__init__()
        self.descriptor = descriptor
        self.type_map = tuple(type_map)
        self.output_key = output_key

        from ase.data import atomic_numbers

        lut = torch.full((max(atomic_numbers.values()) + 1,), -1, dtype=torch.long)
        for symbol, atomic_number in atomic_numbers.items():
            if symbol in self.type_map:
                lut[atomic_number] = self.type_map.index(symbol)
        self.register_buffer("z_to_type", lut, persistent=False)

    def forward(self, data, compute_stress=False, compute_virials=False):
        del compute_stress, compute_virials
        positions = data["positions"]
        batch = data["batch"]
        cells = data["cell"].reshape(-1, 3, 3)
        atomic_types = self.z_to_type[data["atomic_numbers"].long()]
        if bool((atomic_types < 0).any()):
            raise ValueError(f"batch contains elements outside type_map {self.type_map}")

        features = []
        precision = env.GLOBAL_PT_FLOAT_PRECISION
        for frame_index in torch.unique(batch, sorted=True):
            mask = batch == frame_index
            coord = positions[mask].reshape(1, -1, 3).to(precision)
            atype = atomic_types[mask].reshape(1, -1)
            box = cells[frame_index].reshape(1, 9).to(precision)
            ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
                coord,
                atype,
                self.descriptor.get_rcut(),
                self.descriptor.get_sel(),
                mixed_types=self.descriptor.mixed_types(),
                box=box,
            )
            descriptor_output = self.descriptor(
                ext_coord, ext_atype, nlist, mapping
            )[0]
            features.append(descriptor_output.reshape(-1, descriptor_output.shape[-1]))

        data[self.output_key] = torch.cat(features, dim=0).to(positions.dtype)
        return data