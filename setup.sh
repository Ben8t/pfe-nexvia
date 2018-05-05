#!/bin/bash

sudo pip3 install virtualenv
sudo virtualenv -p python3 venv  # create a virtual environment named venv
export PYTHONPATH="$PYTHONPATH:./dev"
source venv/bin/activate
sudo pip3 install -r requirements.txt
deactivate
cd data
if [ "$CIRCLECI" == true ]; then
    wget https://s3.eu-west-2.amazonaws.com/pfe.nexvia/data-min.zip
else
    wget https://s3.eu-west-2.amazonaws.com/pfe.nexvia/data-min.zip
    wget https://s3.eu-west-2.amazonaws.com/pfe.nexvia/data.zip
fi
cd ..