"""
.. _atomistic-tutorial-export:

Exporting a model
=================

.. py:currentmodule:: metatensor.torch.atomistic

This tutorial shows how to define and export an atomistic model following metatensor's
interface.

Model export in metatensor is based on `PyTorch`_ and `TorchScript`_, so make sure you
are familiar with both before reading this tutorial!

.. _PyTorch: https://pytorch.org/
.. _TorchScript: https://pytorch.org/docs/stable/jit.html
"""

# %%
#
# Let's start by importing things we'll need: ``typing`` for the model type annotations,
# ``torch`` itself, the main ``metatensor`` types and classes specific to metatensor
# atomistic models:

import glob
from typing import Dict, List, Optional

import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)


# %%
#
# Defining the model
# ------------------
#
# The model is defined as a class, inheriting from :py:class:`torch.nn.Module`, and
# with a very specific signature for the ``forward()`` function:


class MyCustomModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        pass


# %%
#
# Here ``systems`` will be the list of :py:class:`System` (sometimes also called
# *structures*, or *frames*) for which the model should make a prediction. ``outputs``
# defines what properties should be included in the model output (in case where the
# model supports computing more than one property), as well as some options regarding
# how the properties should be computed in :py:class:`ModelOutput`. ``outputs`` will be
# provided by whoever is using the model: a simulation engine, yourself later, a
# coworker, etc.
#
# Finally, ``selected_atoms`` is also set by whoever is using the model, and is either
# ``None``, meaning all atoms should be included in the calculation, or a
# :py:class:`metatensor.torch.Labels` object containing two dimensions: ``"system"``
# and ``"atom"``, with values corresponding to the structure/atoms indexes to include in
# the calculation. For example when working with additive atom-centered models, only
# atoms in ``selected_atoms`` will be used as atomic centers, but all atoms will be
# considered when looking for neighbors of the central atoms.
#
# Let's define a model that predict the energy of a system as a sum of single atom
# energy (for example some isolated atom energy computed with DFT), and completely
# ignores the interactions between atoms. Such model can be useful as a baseline model
# on top of which more refined models can be trained.


class SingleAtomEnergy(torch.nn.Module):
    def __init__(self, energy_by_atom_type: Dict[int, float]):
        super().__init__()
        self.energy_by_atom_type = energy_by_atom_type

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if list(outputs.keys()) != ["energy"]:
            raise ValueError(
                "this model can only compute 'energy', but `outputs` contains other "
                f"keys: {', '.join(outputs.keys())}"
            )

        # we don't want to worry about selected_atoms yet
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")

        if outputs["energy"].per_atom:
            raise NotImplementedError("per atom energy is not implemented")

        # compute the energy for each system by adding together the energy for each atom
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            for atom_type in system.types:
                energy[i] += self.energy_by_atom_type[int(atom_type)]

        # add metadata to the output
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }


# %%
#
# With the class defined, we can now create an instance of the model, specifying the
# per-atom energies we want to use. When dealing with more complex models, this is also
# where you would actually train your model to reproduce some target energies, using
# standard PyTorch tools.

model = SingleAtomEnergy(
    energy_by_atom_type={
        1: -6.492647589968434,
        6: -38.054950840332474,
        8: -83.97955098636527,
    }
)

# %%
#
# We don't need to train this model since there are no trainable parameters inside. If
# you are adapting this example to your own models, this is where you would train them
# for example like:
#
# .. code-block:: python
#
#    optimizer = ...
#    for epoch in range(...):
#        optimizer.zero_grad()
#        loss = ...
#        optimizer.step()
#
# Exporting the model
# -------------------
#
# Once your model has been trained, we can export it to a model file, that can be used
# to run simulations or make predictions on new systems. This is done with the
# :py:class:`MetatensorAtomisticModel` class, which takes your model and make sure it
# follows the required interface.
#
# When exporting the model, we can define some metadata about this model, so when the
# model is shared with others, they still know what this model is and where it comes
# from.

metadata = ModelMetadata(
    name="single-atom-energy",
    description="a long form description of this specific model",
    authors=["You the Reader <reader@example.com>"],
    references={
        # you can add references that should be cited when using this model here,
        # check the documentation for more information
    },
)

# %%
#
# A big part of exporting a model is the definition of the model capabilities, i.e. what
# are the things that this model can do? First we'll need to define which outputs our
# model can handle: there is only one, called ``"energy"``, which correspond to the
# physical quantity of energies (``quantity="energy"``). This energy is returned in
# electronvolt (``units="eV"``); and with the code above it can not be computed
# per-atom, only for the full structure (``per_atom=False``).


outputs = {
    "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
}

# %%
#
# In addition to the set of outputs a model can compute, the capabilities also include:
#
# - the set of ``atomic_types`` the model can handle;
# - the ``interaction_range`` of the model, i.e. how far away from one particle the
#   model needs to know about other particles. This is mainly relevant for domain
#   decomposition, and running simulations on multiple nodes;
# - the ``length_unit`` the model expects as input. This applies to the
#   ``interaction_range``, any neighbors list cutoff, the atoms positions and the system
#   cell. If this is set to a non empty string, :py:class:`MetatensorAtomisticModel`
#   will handle the necessary unit conversions for you;
# - the set of ``supported_devices`` on which the model can run. These should be ordered
#   according to the model preference.
# - the dtype ("float32" or "float64") that the model uses for its inputs and outputs

capabilities = ModelCapabilities(
    outputs=outputs,
    atomic_types=[1, 6, 8],
    interaction_range=0.0,
    length_unit="Angstrom",
    supported_devices=["cpu"],
    dtype="float64",
)

# %%
#
# With the model metadata and capabilities defined, we can now create a wrapper around
# the model, and export it to a file:

wrapper = MetatensorAtomisticModel(model.eval(), metadata, capabilities)
wrapper.save("exported-model.pt")

# the file was created in the current directory
print(glob.glob("*.pt"))


# %%
#
# Now that we have an exported model, the next tutorial will show how you can use such a
# model to run `Molecular Dynamics`_ simulation using the Atomic Simulating Environment
# (`ASE`_).
#
# .. _Molecular Dynamics: https://en.wikipedia.org/wiki/Molecular_dynamics
#
# .. _ASE: https://wiki.fysik.dtu.dk/ase/
#
