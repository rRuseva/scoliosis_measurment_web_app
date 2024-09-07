from flask import Flask, render_template, request, send_from_directory, url_for, flash, redirect
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from waitress import serve
import os
import image_handler as ih
from typing import Tuple


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ewrfewr'
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
# photos = UploadSet('photos', IMAGES)
# configure_uploads(app, photos)

def allowed_file(file, filename:str) -> Tuple[bool, bool]:
    """Validates if the file is within the allowed files to be uploaded.
    DICOM files not allways have an file extension therefore the file content needs to be checked
    if it has the tag 'DICM'

    Args:
        file (_type_): FileStorage object to be checked
        filename (str): Filename to be checked

    Returns:
        bool: True if the file extension is in the list of the allowed ones or the file content has the tag fot DICOM image.
    """
    if '.' in filename:
        return (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS, False)
    else:
       # check if it is dicom 
        return (ih.validate_is_dicom(file.read()), True)
    

class UploadForm(FlaskForm):
    file = FileField(
        validators=[FileRequired('File field should not be empty')]
    )
    submit = SubmitField('Upload')


@app.route(f'/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# @app.route('/uploads/<name>')
# def get_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    file_url=None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Error no file recieved...')
            return ('Error no file recieved...')
        
        file = form.file.data
        
        # check if empty file without a filename is send
        if file.filename == '':
            flash('No selected file')
            return "No selected file..."
        
        if file:
            is_aallowed, is_dicm = allowed_file(file, file.filename)
            filename = secure_filename(file.filename)
            print(f"filename: {filename}")
            # is_file_allowed =  
            # print(f"is_file_allowed={is_file_allowed}")
            file_url = url_for('get_file', filename=filename)
            print(f"file_url: {file_url}")

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            

            # return redirect(url_for('download_file', name=filename))
            # return f"you file is available <a href='/uploads/{filename}'>here</a>"
        else:
            file_url=None
    return render_template("index.html", form=form, file_url=file_url)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)