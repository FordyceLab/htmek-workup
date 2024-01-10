# HTMEK-Processing
An `.ipynb` interface for HT-MEK data processing.

Summary
-------
This is a python interface for processing HT-MEK data. It is written in python and uses the `ipywidgets` package to create an interactive interface. The interface is designed to be used in a Jupyter notebook. The interface is designed to be used with the HT-MEK data format, but can be used with any data format that is compatible with the `pandas` package.

Installation
------------
To install the package, clone the repository and run the following command in the repository directory:

    pip install -e .
    
Here, the -e flag makes the install "editable" and automatically propagates changes to packages imported in the notebook.

Usage
-----
To use the package, open a Jupyter notebook and run the following command:

    from experiment_processing import processing

This will import the `processing` module. The `processing` module has methods to filter low-quality measurements, plot PDF summaries of HT-MEK data, and export tabular data to a CSV file.


License
-----
This project is licensed under the terms of the MIT license.
