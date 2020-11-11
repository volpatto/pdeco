# PDECO: Study of population dynamics with PDEs for Ecology problems

**READ HERE: This is an early stage research and it should not be considered for reproduction, replication and to draw conclusions.**

## What is it?

This repo constains studies of population dynamics considering both functional responses as well as diffusion effects for Prey-Predator system composed
by Aphids and Ladybird Beetles. [This paper by Banks et al.](https://www.jstor.org/stable/pdf/4218480.pdf?casa_token=UEJ4ZNlRCG4AAAAA:asdZBtFD9_oVwOfAIYHeS-XSQA4wS-M3f2WJ_sl0xvYWzuq284GyfXVRsXfbMSowKlIgDsJbMBG2lQgh2fUefqirJmlltq3Q61FbOLg6jrcH2pQTWaEn) is our main reference.

## How is the mathematical model?

To model the Predator-Prey behavior, we consider a system of partial differential equations (PDEs) based on Lotka-Volterra and others functional response
on the go. This setup can be easily configured in our `jupyter notebook`s provided in this repo.

## How the models are implemented?

We have been using [py-pde](https://github.com/zwicker-group/py-pde), an amazing Python package to solve systems of PDE out-of-the-box. To analyze results,
we use `pandas` and `matplotlib`. Soon model calibration will be implemented, considering both deterministic and bayesian approach.

To document results and for the sake of reproducibility, we use `jupyter notebook`s. We provide a `requirements.txt` file for the case you want to have the same
development environment we have (for instance, you can create an environment with [venv](https://docs.python.org/3/tutorial/venv.html)). We use `Python >= 3.6`.

## License

MIT license. Check LICENSE for further details. Spoiler: you can use it at will, as well as distribute, modify and etc. But without any warranty, at your own risk.

## Authors

* Diego Volpatto: dtvolpatto@gmail.com
* Lucas Dos Anjos: lucas.ecol@gmail.com
