from flask import Flask, render_template, request, send_from_directory
# from weather import get_current_weather
from werkzeug.utils import secure_filename
from waitress import serve
import os

app = Flask(__name__)

upload_folder = os.path.join('static','uploads')

app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)
        return render_template('index.html', image=img_path)
    return render_template('index.html')

# @app.route('/')
# @app.route('/index')
# def index():
#     img_path = os.join.path(upload_folder)
#     return render_template('index.html', image=img_path)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)