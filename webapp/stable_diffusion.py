import os
from .app import create_app
from . import config

app = create_app('default')
