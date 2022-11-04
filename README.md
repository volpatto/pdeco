# PDECO: Study of population dynamics with ODEs for Ecology problems

**READ HERE: This is an early stage research and it should not be considered for reproduction, replication and to draw conclusions.**

Try it out in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Tungdil01/pdeco/HEAD)

## What is it?

This repo constains studies of population dynamics considering both functional responses as well as diffusion effects for Prey-Predator system composed of Aphids and 
Ladybeetles. [This paper by Banks et al.](https://dx.doi.org/10.1007/BF00378930) and the paper of [Lin and Pennings](https://dx.doi.org/10.1002/ece3.4117) are our main 
references.

## How is the mathematical model?

To model the Predator-Prey behavior, we consider a system of ordinary differential equations (ODEs) based on Lotka-Volterra and others functional response
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
* Gustavo Naozuka: gtnaozuka@gmail.com
