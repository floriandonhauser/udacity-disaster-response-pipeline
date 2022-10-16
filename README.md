# Disaster Response Pipeline Project

### Dependencies to install with pip
Install all required packages from requirements.txt. `pip install -r requirements.txt`


### Project Overview
This project is a web app which can be used to classify a message from a disaster event into several categories.  
It uses a machine learning model to detect which of the 36 classes the message could belong to.  
Using such an approach, messages could be forwarded to the right agency when a new disaster message is received so that the correct measures can be taken.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Files:
<pre>
.
├── app
│   ├── run.py------------------------# SCRIPT TO RUN THE FLASK APP
│   └── templates
│       ├── go.html-------------------# WEBSITE TO SHOW CLASSIFICATION RESULTS
│       └── master.html---------------# MAIN WEBSITE
├── data
│   ├── DisasterResponse.db-----------# DATABASE
│   ├── disaster_categories.csv-------# CATEGORIES TO PROCESS
│   ├── disaster_messages.csv---------# MESSAGES TO PROCESS
│   └── process_data.py---------------# SCRIPT FOR ETL PIPELINE
├── models
│   └── train_classifier.py-----------# SCRIPT TO CREATE AND TRAIN ML MODEL

</pre>
