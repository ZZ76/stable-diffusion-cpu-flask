import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:

    USE_MODEL = False or os.environ.get('USE_MODEL')

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    pass

config = {
        'default': DevelopmentConfig
        }
