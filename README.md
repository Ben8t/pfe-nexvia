# PFE - Nexvia
## Reverse engineering of a determinist model

The main objective of this project is to recover the modeling of the calculations of
acquisition costs when buying a property. From a base of training built on a deterministic model our goal will be to find the modeling
from machine learning algorithms (reverse engineering).

You can find the slideshow for this project [here](report/slideshow.pdf)

Some parts of this project have been purposely deleted for privacy reasons.

## Setup and details 
* Main scheme : https://goo.gl/hd1Hdr

### Install
* Clone the project with `git clone url`.
* Run `. setup.sh`.

### Get started

* Run the virtualenv with `. venv/bin/activate`

Within python or ipython shell you can execute commands below to test a basic worflow :

* **Basic workflow** :

```
from dev.processing.data_processing import *
from dev.processing.extract import *
from dev.processing.load_data import *
from dev.analysis.analysis import *
from dev.analysis.error import *
from dev.model.model import *
import pandas
import seaborn
import matplotlib.pyplot as pyplot

# Loading raw data
data = load_data()

# Cleaning data (remove na if needed, remove undesired output, etc...)
cleaned_data = cleaning_data(data)

# Processing data (encode qualitative data, select features, etc...)
processed_data = processing_full(cleaned_data)

# Split data set intro train and test sets
x_train, y_train , x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")

# Linear Regression (easily modifiable, for exampe with "model_random_forest" for a Random Forest)
model = model_regression(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Print score performances
error_print_score(x_test, y_test, y_pred)
```

### Use `run.py` interface

* Launch `python run.py` to get into the interface.

* `run.py` functional scheme : https://goo.gl/9KgdKF


An end of studies' project made by Aymeric Duchein, Benoit Pimpaud, Jordan Rostren and Paul Schaeffer.
