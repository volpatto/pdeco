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

We have applied Bayesian calibration for the model's parameters.

To document results and for the sake of reproducibility, we use `jupyter notebook`s. We provide a `requirements.txt` file for the case you want to have the same
development environment we have (for instance, you can create an environment with [venv](https://docs.python.org/3/tutorial/venv.html)). We use `Python >= 3.6`.

## How to use (and modify) the code?

Open the "notebooks" directory. You will find three codes .ipynb:

- phenomenological.ipynb
This code contains the routines for the three employed phenomenological models. The routine includes (i) loading the data, (ii) data regularisation, (iii) sensitivity analysis, and (iv) Bayesian calibration.

- sindy_sa.ipynb
This code contains the routines for the three employed phenomenological models. The routine includes the same functionalities as the phenomenological code.

- ModelSelection.ipynb
This code performs the model selection after getting the results from the two previous codes. Each model has Maximum a Posteriori (MAP) estimates for the calibrated parameters. We manually insert the values in the model, for example for model (1):
dRdt = [ r1 * R - a1 * R * C / ( 1 + a2 * R ),
	ef * R - m * C * C ]
it becomes:
dRdt = [ 0.318752 * R - 0.080779 * R * C / ( 1 + 0.0001948159776916025 * R ),
	0.0002897334690674872 * R - 0.012509 * C * C ]
After filling in for all models, 

To test another mathematical model the user should directly change the definition of an existing model, that is indicated by the prefix "def" (which is a function of Python). All the parameters should also be modified througour the code to be consistent with the new model to be added.

## License

MIT license. Check LICENSE for further details. Spoiler: you can use it at will, as well as distribute, modify and etc. But without any warranty, at your own risk.

## Authors

* Diego Volpatto: dtvolpatto@gmail.com
* Lucas Dos Anjos: lucas.ecol@gmail.com
* Gustavo Naozuka: gtnaozuka@gmail.com
