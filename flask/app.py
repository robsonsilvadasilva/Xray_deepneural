from flask import Flask, request, render_template
import os
from commons import get_tensor
from inference import get_xray_predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
       
     
    if request.method == 'POST':
        if 'file' not in request.files:
            print("File not uploaded")
            return 
        file = request.files['file']
        image = file.read()
        pred = get_xray_predict(image_bytes=image)
        return render_template('result.html', predi=pred)


if __name__ == '__main__':
    app.run(debug=True)