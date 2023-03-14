# PDECO: Study of population dynamics with ODEs for Ecology problems

Try it out in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Tungdil01/pdeco/HEAD)

## What is it?

This repo contains studies of population dynamics considering a Prey-Predator system composed of Aphids and Ladybeetles, as a case study with the data from the paper of [Lin and Pennings](https://dx.doi.org/10.1002/ece3.4117).

## How is the mathematical model?

To model the Predator-Prey behavior, we consider a system of ordinary differential equations (ODEs) based on a structure of Lotka-Volterra. This setup can be easily configured in our `jupyter notebook`s provided in this repo.

## How the models are implemented?

We have applied Bayesian calibration for the model's parameters.

To document results and for the sake of reproducibility, we use `jupyter notebook`s. We provide a `requirements.txt` file in case you want to have the same
development environment we have (for instance, you can create an environment with [venv](https://docs.python.org/3/tutorial/venv.html)). We use `Python >= 3.6`.

## Required packages
For the utilized packages, see the [requirements](https://github.com/Tungdil01/pdeco/blob/main/notebooks/requirements.txt).

## How to use (and modify) the code?

Open the _notebooks_ folder. You will find seven codes .ipynb:

- model1.ipynb to model3.ipynb
These codes contain the routines for the three employed phenomenological models (model 1, 2, and 3). The routine includes (i) loading the data, (ii) data regularisation, (iii) sensitivity analysis, and (iv) Bayesian calibration.

- model4.ipynb to model6.ipynb
These codes contain the routines for the three data-driven models (model 4, 5, and 6). The routine includes (i) loading the data, (ii) data regularisation, (iii) sensitivity analysis, and (iv) Bayesian calibration.

- ModelSelection.ipynb
This code performs the model selection after getting the results from the two previous codes. Each model has Maximum a Posteriori (MAP) estimates for the calibrated parameters. We manually insert the values in the model, for example for model (1):

R, C = N

dNdt = [ r1 * R - a1 * R * C / ( 1 + a2 * R ),
 ef * R - m * C * C ]
 
it becomes:

dNdt = [ 0.318752 * R - 0.080779 * R * C / ( 1 + 0.0001948159776916025 * R ),
 0.0002897334690674872 * R - 0.012509 * C * C ]
 
After filling in all models, the user can run the code and will get a ranking from worst to best.

To test another mathematical model not listed, the user should directly change the definition of an existing model, which is indicated by the prefix "def" (this is a function of Python). All the parameters should also be modified throughout the code to be consistent with the new model to be added.

### How to run the data-driven model identification codes?

Open a terminal and type:

1. Open the _data_driven_model_identification_ folder. You will find two subfolders whose scripts are used to discover models 4, 5, and 6. The identification of models 4 and 6 uses the same scripts but different GPR (Gaussian Process Regression) kernels for the data regularisation.

 2. Enter the directory of the model you would like to run.

 3. Clean the project's _output_ folder:
	 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ make clean`

 4. Run the project by writing the output logs to the terminal:
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ make`
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**OR**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run the project by writing the output logs to the _models.dat_ file:
	 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ make run_output`

 5. Enter the project's _output_ folder and check out the results.

## License

MIT license. Check LICENSE for further details. Spoiler: you can use it at will, as well as distribute, modify and etc. But without any warranty, at your own risk.

## Authors

* Diego Volpatto: dtvolpatto@gmail.com
* Lucas Dos Anjos: lucas.ecol@gmail.com
* Gustavo Naozuka: gtnaozuka@gmail.com
