from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import io
from model import create_model
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 加载模型
model = create_model()
model.load_state_dict(torch.load('model_50.0.pt', map_location=torch.device('cpu')))
model.eval()

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
    _, predicted = outputs.max(1)
    return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            img_bytes = file.read()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            with open(file_path, 'wb') as f:
                f.write(img_bytes)
            prediction = get_prediction(img_bytes)
            return render_template('index.html', prediction=prediction, image_url=file.filename)
    return render_template('index.html', prediction=None, image_url=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
