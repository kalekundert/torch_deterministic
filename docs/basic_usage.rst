***********
Basic Usage
***********

Installation
============
Install ``torch_deterministic`` using ``pip``::

    $ pip install torch_deterministic

- Requires python≥3.8
- `Semantic versioning`_.

.. _`semantic versioning`: https://semver.org/

Example Dataloader
==================

Make a dataset that uses a pseudorandom number generator (PRNG) seeded on the 
index number to generate augmentations:

.. code-block:: python

    import numpy as np

    class MyDataset:

        def __getitem__(self, i):
            rng = np.random.default_rng(i)
            return rng, rng.uniform()

Configure a data loader to sample from this dataset:

.. code-block:: python

    import torch_deterministic as td
    from torch.utils.data import DataLoader

    dataset = MyDataset()
    sampler = td.InfiniteSampler(
        epoch_size=1000,
        shuffle=True,
        increment_across_epochs=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=td.collate_rngs,
    )
