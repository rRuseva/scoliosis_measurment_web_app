from flask import Flask, render_template, request, send_from_directory, url_for, flash, redirect
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from waitress import serve
import os
import image_handler as ih

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ewrfewr'
app.config['UPLOAD_FOLDER'] = 'uploads'

# photos = UploadSet('photos', IMAGES)
# configure_uploads(app, photos)
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', '']

def allowed_file(filename):
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    else:
        ih.parse_image_file(filename=filename)
    return True
    # return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# class UploadForm(FlaskForm):
#     photo = FileField(
#         validators=[# FileAllowed(photos, 'Only images are allowed'),
#                     FileRequired('File field should not be empty')
#         ]
#     )
#     submit = SubmitField('Upload')

# @app.route(f'/uploads/<filename>')
# def get_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
from flask import send_from_directory

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    file_url=None
    if request.method == 'POST':
        # check if the post request has the file part
        print("request.files", request.files)
        if 'file' not in request.files:
            flash('No file part')
            return ('Error no file recieved...')
        print("request: ", request)
        file = request.files['file']
        print("file:",file) 
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return "No selected file..."
        if file and allowed_file(file.filename):
            print("file.filename: ",file.filename)
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('download_file', name=filename)

            # return redirect(url_for('download_file', name=filename))
            return f"you file is available <a href='/uploads/{filename}'>here</a>"
        else:
            file_url=None
    print(file_url)
    return render_template("index.html")

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)