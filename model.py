import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib  # Hoặc import pickle nếu bạn muốn dùng pickle

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('WELFake_Dataset.csv')

# Kiểm tra dữ liệu
print(df.head())

# Xử lý giá trị NaN trong các cột văn bản
df['title'] = df['title'].fillna('')  # Thay thế NaN bằng chuỗi rỗng trong cột 'title'
df['text'] = df['text'].fillna('')    # Thay thế NaN bằng chuỗi rỗng trong cột 'text'

# Kết hợp 'title' và 'text' thành một cột duy nhất
df['combined_text'] = df['title'] + ' ' + df['text']

# Chia dữ liệu thành đặc trưng và nhãn
X = df['combined_text']
y = df['label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng pipeline với TfidfVectorizer và LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Chuyển đổi văn bản thành đặc trưng số
    ('clf', LogisticRegression())  # Mô hình Logistic Regression
])

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Lưu mô hình vào tệp model.pkl
joblib.dump(pipeline, 'model.pkl')

print("Model đã được lưu vào 'model.pkl'")
