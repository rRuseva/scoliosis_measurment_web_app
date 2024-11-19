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
app.config['TEMP_FOLDER'] = 'temp'

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
        file_content = file.read()
        file.seek(0)
        return (ih.validate_is_dicom(file_content), True)
    

def clear_temp_folder():
    temp_files = os.listdir(app.config['TEMP_FOLDER'])
    if len(temp_files) > 0:
        for filename in temp_files:
            os.remove(os.path.join(app.config['TEMP_FOLDER'], filename))


class UploadForm(FlaskForm):
    file = FileField(
        validators=[FileRequired('File field should not be empty')]
    )
    submit = SubmitField('Upload')


@app.route(f'/uploads/<filename>')
def get_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route(f'/temp/<filename>')
def get_processed(filename):
    print(f"processed images: filename={filename}")
    return send_from_directory(app.config['TEMP_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    file_url=None
    result_images = []
    result_cob_angles = []
    result_apex = []
    filename = ' '
    clear_temp_folder()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Error no file received...')
            return ('Error no file received...')
        
        file = form.file.data
        # check if empty file without a filename is send
        if file.filename == '':
            flash('No selected file')
            return "No selected file..."
        
        if file:
            is_allowed, is_dicm = allowed_file(file, file.filename)
            filename = secure_filename(file.filename)
            file_url = url_for('get_upload_file', filename=filename)
            print(f"Uploading {filename} to {file_url}...")
            if is_allowed and not is_dicm:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if is_allowed and is_dicm:
                file.seek(0)
                file_content = file.read()
                ih.save_dicom(os.path.join(app.config['UPLOAD_FOLDER'], filename), file_content)

            # result_url =  url_for('get_temp_file', filename=filename)
            # file_url = url_for('get_temp_file', filename=filename)
            if is_allowed:
                result = ih.process_image(filename, app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER'])
                result_cob_angles = [round(res[2], 5) for res in result ]
                result_apex = [(round(res[0], 5) , round(res[1], 5) ) for res in result ]

                for filename in os.listdir(app.config['TEMP_FOLDER']):
                    result_images.append( (filename, url_for('get_processed', filename=filename)))
                    print(filename)
            
        else:
            file_url=None
            print(f"File {filename} not uploaded")
    return render_template("index.html",
                           form=form,
                           file_url=file_url,
                           images=result_images,
                           cob_angles=result_cob_angles,
                           apices = result_apex,
                           number_of_angles=len(result_cob_angles),
                           original_image_name=filename.split('.')[0])

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)