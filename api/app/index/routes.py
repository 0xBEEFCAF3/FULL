from flask import Blueprint, render_template, request
from random import randint


home = Blueprint('index', __name__)


@home.route('/', methods=['GET'])
def index():
    return render_template('home.html', title='Home')
