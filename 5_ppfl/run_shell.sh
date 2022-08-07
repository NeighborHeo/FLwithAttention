#!/bin/bash

python3 manage.py runserver
# python3 custom_client.py --edge -1 --local True
python3 custom_client.py --edge 0
python3 custom_client.py --edge 1
python3 custom_client.py --edge 2
