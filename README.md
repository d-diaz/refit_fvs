Refit FVS
=========

Tuning Parameters of the Forest Vegetation Simulator

This project is designed to re-fit fundamental growth and mortality equations involved in simulating
forest growth-and-yield using the Forest Vegetation Simulator (FVS) against field data. 

The model-tuning functionality developed here is intended to support the re-fitting of fundamental
FVS equations using the most recent version of the US Forest Service's Forest Inventory & Analysis
(FIA) datasets for a given ecoregion or set of regions. In theory, this re-fitting or calibration
process can be applied with any dataset the user provides, so long as the input database and FVS
keyfile(s) required to execute simulations with FVS and compare observed vs. modeled forest growth
and mortality are prepared and formatted appropriately.



Project Organization
--------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources in original formats.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- Data sets ready for modeling.
    │   └── raw            <- Extracted and/or unprocessed data ready for processing.
    │
    ├── docs               <- Read The Docs documentation.
    │
    ├── models             <- Outputs from model calibration and tuning.
    │
    ├── notebooks          <- Jupyter notebooks for data processing, modeling, and interpretation.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml   <- The requirements for reproducing the analysis environment,
    │                         installable as a conda environment.
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │                     
    ├── refit_fvs          <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── models         <- Scripts to train models use them to make predictions.
    │   │
    │   ├── visualization  <- Scripts to visualize and interpret data and models.
    │   │
    │   └──  tests         <- Scripts to test source code.
    │
    └── .travis.yml        <- configuration file for continuous integration with Travis CI.


--------

<p><small>Repository organization adapted from the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
