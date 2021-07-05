# Language Technology Project 2021

We present a NLP pipeline targeted at machine-translation to human-translation comparison.
The Europarl dataset is used.

## Requirements

Instructions are targeted towards Debian/Ubuntu based systems.
Consult the documentation at <https://docs.python.org/3/library/venv.html> for handling virtual environments on other operating systems.

Firstly, install Python, pip and venv.

    sudo apt install python3 python3-pip python3-venv

Second, navigate to the project's root directory.

    cd ~/path/to/project

Initialize a virtual environment called venv using the `venv` module.

    python3 -m venv venv

Activate the virtual environment env.

    source venv/bin/activate

Install the required project modules into the virtual environment.

    pip3 install -r requirements.txt

You may wish to test that `torch` in particular was correctly installed.

    python3 -c "import torch; x = torch.rand(5, 3); print(x)"

If you are using a dedicated GPU you may wish to evaluate whether it is accessible by `torch`.

    python3 -c "import torch; print(torch.cuda.is_available())"

After using the pipeline you may wish to deactivate the virtual environment as follows.

    deactivate

The `benepar` and `spacy` modules require additional data that is downloaded as follows.

    python3 setup.py

## Running

Download the datasets from here (too big for github):

    https://drive.google.com/file/d/1X4VSUonqtCCtSIByR9S3nvEZL2knn61M/view?usp=sharing

Download and unpack the folder `data copy`, rename it to `data`, and replace the `data` folder in this project. On Linux the following is equivalent.

    cd ~/Downloads
    unzip data.zip
    mv -v "data copy"/* path/to/project/data

The outcome should look as follows.

    project/
        data/
            DATA.md
            dev/
            test/
            train/

Navigate to the project's root directory.

    cd ~/path/to/project

Execute the `classifier` module for the feature based SVM.

    python3 src/classifier.py

Execute the `bert` module for the Bert network.
The `--override` option overrides model data from previous runs; passing this flag will usually be necessary when training a new model.
Use flag `--nrows` to control the amount of data loaded.
use `--epochs` to control the number training of epochs.

    python3 src/bert.py --override

## Architecture

Go to [ARCHITECTURE.md](ARCHITECTURE.md) to learn more about this project's architecture.

## Contributing

Go to [CONTRIBUTING.md](CONTRIBUTING.md) to learn more about contributing to this repository.

## Data

Go to [DATA.md](data/DATA.md) to learn more about loading data into this repository; this will be necessary to run the pipeline.
