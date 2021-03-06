# FAIR-FATE

by
Teresa Salazar,
Miguel Fernandes,
Helder Araujo,
Pedro Henriques Abreu

## General Structure

All source code used to generate the results and figures in the paper are in
the `code` folder.
The datasets used in this study are provided in `datasets`.
Results generated by the code are saved in the respective dataset folder.

## Dependencies

You'll need a working Python environment to run the code.
The required dependencies are specified in the file `requirements.txt`.

You can install all required dependencies by running:

    pip install -r requirements.txt

## Reproducing the results

To build and test the software and produce all results run this in the top level of the repository:

    sh script.sh
    
## Running the code with your own dataset

To build and test the software and produce all results on your own dataset you will need to:

- create a new file on the `datasets` folder with the name of your dataset
- create a Model (following, for example, the file Adult.py)
- add the model in the `run.py` file in which given the dataset name a new Model is generated.

## License

All source code is made available under a Creative Commons License license. (https://creativecommons.org/licenses/by/4.0/)
