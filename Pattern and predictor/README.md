# Long-Term-Energy-Forecasting
LTEF: Long-Term Energy Forecasting

This code is divided into two modules: Energy pattern model (1) and Energy forecasting model (2):

1) Energy pattern model:

This code attempts to extract seasonal patterns from historical data on electricity production. The output data depends on the user's needs and the raw data frequency (minutes, hours, days, etc.)

The process is as follow:

- Reading a input file and check if the pattern file has already been created.
- Loading the data for further processing.
- Training the Prophet machine learning model to generate trends and series decomposition.
- Generating output data from Prophet model and storing the results in a CSV file for later use.

2) Energy forecasting model

This module creates long term forecast for each energy source thanks to the pattern model. 

The process is as follow:

- Reads original and previously formatted data to generate a data line between the database and a target value (defined in input file)
- Uses pattern file from patterns module to adjust the data line to values with seasonal trends and their inputs/outputs defined by the user.
- Stores prediction results in a database for further analysis





## Installation
- The code runs with Anaconda + Prophet # (pip install prophet)
- Optionally:
  - Install python 3.10 or 3.9
  - Install requirements.txt with pip install -r requirements.txt

## Pattern Module

- Run the program with python Pattern.py your_input.yml  #(i.e. Chile.yml)
- Results are shown in a csv file called Percentage.csv


## Predictor Module

- Run the program with python Predictor.py your_input.yml  #(i.e. Chile.yml)
- Results are shown in a csv file called Forecast.csv


## Database format:

- Demand: filename = Demand.csv (per day)
 	Date,demand
 	yy-mm-dd, production
- Production: filename as user choice (per day)
	Date, source1, source2,...
        yy-mm-dd, production





