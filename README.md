# Embedded ML Ear PPG

This repository contains the code related to the 
[End-to-End Embedded Machine Learning for In-Ear PPG Peak Detection](https://repository.tudelft.nl/record/uuid:994a13d2-093c-41b0-84df-c7d254e936c0)
thesis paper.

This work establishes a standardised framework for automatically identifying optimal embedded model architectures\
for in-ear PPG analysis.

## Pre-requisites

Before running the code, ensure you have a virtual environment set up with the required dependencies.
You can create a virtual environment and install the dependencies using the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

This repository contains two distinct parts:

- Data pre-processing visualization tool
- Embedded ML model training and evaluation framework

Do note that for each part a self-supplied dataset is required to run
the code. The dataset used in the thesis is not included in this repository,
due to confidentiality restrictions.

### Data Pre-processing Visualization Tool

The data pre-processing visualization tool is located in the `plot_files/align_signals_app` directory.
This tool allows users to visualise and verify the alignment of PPG signals.

To run the visualization tool, navigate to the `plot_files/align_signals_app` directory and execute the following command:

```bash
python app.py
```


### Embedded ML Model Training and Evaluation Framework

The framework can be started by running the `main.py` file located in the root directory of the repository.
To see the available options and configurations, you can run the following command:

```bash
python main.py --help
```





