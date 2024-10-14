#!/bin/bash


>&2 echo "Checking virtual env"

if [[ "$VIRTUAL_ENV" == "" ]]
then
    >&2 echo "Please run from inside a virtual environment"
    exit 1
fi

>&2 echo "OK"

>&2 echo "Ensuring requirements are met"

pip install -r requirements.txt > /dev/null
REQ=$?

if [ $REQ -ne 0 ]
then
    >&2 echo "Error(s) when installing requirements"
    exit 1
fi

>&2 echo "OK"


python -m app


