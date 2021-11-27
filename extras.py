#!/usr/bin/env python
# -*- coding: utf-8 -*-import json

import logging
import yaml
import os
import sys
import json


def get_config():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_directory)
    with open(f'{current_directory}/config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    return config
