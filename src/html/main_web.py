import os
from flask import Flask, render_template, request, url_for
 

IMAGE_SOURCE = os.path.join('static','images')
app = Flask(__name__)
app.config['UPLOADED_PHOTO'] = IMAGE_SOURCE

@app.route("/a")
def index(name=None):
    logo_image = os.path.join(app.config['UPLOADED_PHOTO'], 'logo.png')
    return render_template("index.html", name=name, logo_image = logo_image)





if __name__ == '__main__':
    app.run(debug=True)
    