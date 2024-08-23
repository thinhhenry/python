from flask import Flask, render_template, request
import joblib  # Hoặc import pickle nếu bạn muốn dùng pickle

# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình đã lưu
model = joblib.load('model.pkl')

# Định nghĩa trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title')
    text = request.form.get('text')

    # Kết hợp title và text
    combined_text = f"{title} {text}"

    # Dự đoán
    prediction = model.predict([combined_text])[0]

    # Chuyển đổi dự đoán thành chuỗi hiển thị
    result = 'Real' if prediction == 1 else 'Fake'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
