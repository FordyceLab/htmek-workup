# HTMEK-workup
HTMEK-workup provides scripts and an `.ipynb` interface to work up kinetic data from HT-MEK experiments.

Summary
-------
This is a python interface for processing HT-MEK data. It is written in python and uses `ipywidgets` elements to provide an interactive interface in a compatible environment (e.g. Jupyter, IDE renderer). The interface is designed to be used with the HT-MEK data format, but can be used with any data format that is compatible with the `pandas` package.

Usage
-----
To use the package, clone the repository via GitHub desktop or from the terminal with the following command:

    git clone https://github.com/FordyceLab/htmek-workup.git

Then, open one of the example `ipynb` and import the processing scripts from the main directory:

    from experiment_processing import processing

This will import the `processing` module. The `processing` module includes methods for fitting and filtering data, plotting PDF summaries of HT-MEK data, and exporting measured parameters as tabular data to a CSV file.


License
-----
This project is licensed under the terms of the MIT license.
