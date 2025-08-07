from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from celery import Celery
import redis
from project.config import Config
import logging
from logging.handlers import RotatingFileHandler
from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity
from datetime import timedelta
import os

log_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logging.basicConfig(level=Config.LOG_LEVEL, handlers=[log_handler])

app = Flask(__name__)

origins = Config.ALLOWED_CORS_ORIGINS.split(',')
CORS(app, resources={r"/*": {"origins": origins}})

app.config.update(
    CELERY_BROKER_URL=Config.CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND=Config.CELERY_RESULT_BACKEND
)

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

r = redis.Redis.from_url(Config.CELERY_BROKER_URL, decode_responses=True)

socketio = SocketIO(app,
                    async_mode='eventlet',
                    cors_allowed_origins=origins,
                    engineio_logger=Config.LOG_LEVEL == 'DEBUG')

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)

def authorize_user_for_ticker(user_id, ticker):
    # Implementierung je nach Geschäftslogik
    return True

@socketio.on('connect')
def handle_connect():
    try:
        verify_jwt_in_request()
        current_user = get_jwt_identity()
        logger.info(f"Authenticated WebSocket connection from {current_user}")
    except Exception as e:
        logger.warning(f"Unauthorized WebSocket connection attempt: {str(e)}")
        return False

@socketio.on('realtime_subscribe')
def handle_realtime_subscribe(data):
    try:
        verify_jwt_in_request()
        current_user = get_jwt_identity()
        if not authorize_user_for_ticker(current_user, data['ticker']):
            emit('error', {'message': 'Unauthorized for this ticker'})
            disconnect()
    except Exception as e:
        emit('error', {'message': 'Authentication failed'})
        disconnect()