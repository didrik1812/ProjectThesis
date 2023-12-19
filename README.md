# ProjectThesis
This repo contain code that is used in my project thesis "Genotype to Phenotype Predictions Using Boosting Algorithms", and it is mostly meant to show the methods I have used. The data that I have used is not made available here, only the processing of the data is included.
## How to replicate results
**Requirements:** Python 3.8, anaconda and R of version 4.2+
1. Clone Repo
2. Create conda environment `projectThesis` with necessary packages by running `conda env create -f environment.yml`.
3. Edit the configuration script `code/model_exploration.py` to your needs (described in file) and run it. 

## Description of Files in `code/`
- `testModel.py`: This file contains the definition of the class `testModel` that performs cross-validation and hyperparmater tuning with the selected model on the chosen data. The procedure is illustrated below (click on picture, or change to light mode if you are in dark mode to see the image properly). ![CVprocedure drawio(1)](https://github.com/didrik1812/ProjectThesis/assets/92478930/a0ee92d4-b198-43a7-94cf-9a9bc04ee453)
- `model_exploration.py`: Is the "configuration script", change global variables here to use the `testModel` class for your choosen model.
- `INLAcv.R`: Runs a Bayesian animal model fitted with INLA using a 10 fold cross-validation.
- `dataloader.R` and `envGenedataloader.R`: Scripts that prep data for the two-step and the one-step approach, respectively. Based on code from Stefanie Muff
- `h_dataPrep.R`: Helper script for the dataloader scripts, made available by Stefanie Muff.
