# Dataset acquisition and qualification

The primary target is the authors' `cace-lr-fit` release associated with the
LES paper.
The local release mirror has a `fit-TiO2-NaCl-aq` directory containing
`equil.xyz`, `best_model.pth`, a conversion script, and a DeepMD export.
The export has one frame with 8,784 atoms and species H/O/Na/Cl/Ti.
The corresponding Zenodo record is `10.5281/zenodo.16415060` (V0, published
July 24, 2025), with the open-access file record linked as
`10.5281/zenodo.16415061`.

The release should be obtained from the authors' Zenodo record or its linked
repository rather than inferred from the example directory.

## Acquisition order

1. Download the tagged `cace-lr-fit` release or its Zenodo archive and inspect
   the files under `fit-TiO2-NaCl-aq`.
2. Treat `equil.xyz` as a candidate trajectory only after checking whether it
   contains energies, forces, cells, atom types, and units in its extended-XYZ
   records.
3. Prefer the complete labeled trajectory or source data used by the campaign.
4. If only structures or a checkpoint are distributed, contact the authors
   before treating the data as a training source.
5. Keep the downloaded archive outside the public repository and record its
   DOI, version, checksum, and license in the campaign manifest.

## Qualification gate

The dataset is accepted only if the inspection confirms:

- coordinates and periodic cells for every frame;
- energies and forces from one documented electronic-structure protocol;
- atom types covering the oxide, water, and electrolyte species;
- enough independent frames for train, validation, and held-out test splits;
- more than one electrolyte or interfacial environment, or a documented
  trajectory that samples materially different dielectric response;
- units, atom ordering, and boundary conventions that can be converted to the
  project's DeepMD layout without guessing;
- permission to use the data for the planned campaign.

The local `equil.xyz` record does contain a periodic cell, energy, forces, and
species, and the converted labels are aligned.
However, `deepmd/manifest.txt` records that it is a single frame and that the
training trajectory (`../TiO2-electrolyte/train.xyz`) is absent from the
release mirror.
It therefore fails the independent train/validation/test requirement and must
not be used for formal training without obtaining the missing trajectory.
The public `best_model.pth` is a model artifact, not a substitute for labeled
training data.

## Fallback candidates

If the complete TiO2-NaCl-aq labels are unavailable, evaluate the public
TiO2-water DeepMD data as a tooling and conversion fallback.
It is useful for oxide-water interfaces, but it does not by itself establish
the desired salt and dielectric-environment comparison.

Do not promote a fallback to the formal benchmark until its missing physical
coverage is explicitly accepted and documented.
