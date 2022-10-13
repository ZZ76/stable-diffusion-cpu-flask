from flask import Flask
from flask_bootstrap import Bootstrap5
from flask_moment import Moment
from ..config import config

bootstrap = Bootstrap5()
moment = Moment()

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    bootstrap.init_app(app)
    moment.init_app(app)
    
    if app.config['USE_MODEL'] == 'true':
        from ... import generator
    else:
        from ... import generator_test as generator
    # print('Loading from create_app')
    app.generator = generator.generator()
    app.generator.load_model()

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    

    return app
