import streamlit as st
import numpy as np
import pickle

# Hàm để tải mô hình và các đặc trưng đã chọn
def load_models_and_features():
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

# Hàm để dự đoán bệnh tim
def predict_heart_disease(model, input_data, selected_features_indices):
    input_array = np.array(input_data).reshape(1, -1)
    input_array_selected = input_array[:, selected_features_indices]
    prediction = model.predict(input_array_selected)
    return prediction[0]

# Thiết lập tiêu đề và hình ảnh tiêu đề
st.set_page_config(page_title="Dự đoán Bệnh Tim", page_icon="❤️", layout="wide")
st.title("🩺 Dự đoán Bệnh Tim")
st.markdown("Sử dụng mô hình học máy để dự đoán nguy cơ mắc bệnh tim.")

# Tải mô hình và các đặc trưng đã chọn
models = load_models_and_features()
selected_features_indices = models["selected_features_indices"]

# Lựa chọn mô hình
model_choice = st.sidebar.selectbox("Chọn mô hình", 
                                    ["Logistic Regression", 
                                     "SVM", 
                                     "Tuned SVM"])

# Tải mô hình tương ứng
model = None
if model_choice == "Logistic Regression":
    model = models['logistic_regression']
elif model_choice == "SVM":
    model = models['svm']
elif model_choice == "Tuned SVM":
    model = models['svm_tuned']

# Tạo các input cho người dùng nhập liệu
st.sidebar.title("Chọn đầy đủ thông tin trước khi dự đoán!")

# Sử dụng cột để bố trí các input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Tuổi", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
    cp = st.selectbox("Loại đau ngực", options=[0, 1, 2, 3])
    trestbps = st.number_input("Huyết áp tâm thu khi nghỉ (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Mức độ Cholesterol trong máu (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Đường huyết lúc đói", options=[0, 1], format_func=lambda x: "> 120 mg/dl" if x == 0 else "<= 120 mg/dl")

with col2:
    restecg = st.selectbox("Kết quả điện tâm đồ lúc nghỉ", options=[0, 1, 2])
    thalach = st.number_input("Nhịp tim tối đa đạt được khi gắng sức", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Đau thắt ngực do gắng sức", options=[0, 1])
    oldpeak = st.number_input("Sự chênh lệch ST khi gắng sức so với lúc nghỉ ngơi", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("Độ dốc của đoạn ST khi gắng sức", options=[0, 1, 2], format_func=lambda x: "Tăng dần" if x == 0 else "Phẳng" if x == 1 else "Giảm dần")
    ca = st.selectbox("Số lượng mạch chính bị hẹp", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Tình trạng Thalassemia", options=[0, 1, 2, 3])

if st.sidebar.button("Kết quả dự đoán"):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    if model is not None:
        prediction = predict_heart_disease(model, input_data, selected_features_indices)
        if prediction == 1:
            st.success("Nguy cơ mắc bệnh tim.")
        else:
            st.success("Không có nguy cơ mắc bệnh tim.")
    else:
        st.error("Vui lòng chọn một mô hình hợp lệ.")
