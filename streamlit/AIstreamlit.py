import streamlit as st
import numpy as np
import pickle

# Hàm để tải mô hình đã lưu
def load_model(model_name):
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model

# Hàm để tải các đặc trưng đã chọn
def load_selected_features():
    with open('selected_features.pkl', 'rb') as f:
        selected_features_indices = pickle.load(f)
    return selected_features_indices

# Hàm để dự đoán bệnh tim
def predict_heart_disease(model, input_data, selected_features_indices):
    input_array = np.array(input_data).reshape(1, -1)
    input_array_selected = input_array[:, selected_features_indices]
    prediction = model.predict(input_array_selected)
    return prediction[0]

st.title("Dự đoán bệnh tim")

# Lựa chọn mô hình
model_choice = st.selectbox("Chọn mô hình", 
                            ["Logistic Regression", 
                             "Decision Tree", 
                             "SVM", 
                             "Best Logistic Regression", 
                             "Best SVM", 
                             "LightGBM"])

# Tải mô hình tương ứng
model = None
if model_choice == "Logistic Regression":
    model = load_model('logistic_model.pkl')
elif model_choice == "Decision Tree":
    model = load_model('decision_tree_model.pkl')
elif model_choice == "SVM":
    model = load_model('svm_model.pkl')
elif model_choice == "Best Logistic Regression":
    model = load_model('best_logistic_model.pkl')
elif model_choice == "Best SVM":
    model = load_model('best_svm_model.pkl')
elif model_choice == "LightGBM":
    model = load_model('lgbm_model.pkl')

# Tải danh sách các đặc trưng đã chọn
selected_features_indices = load_selected_features()

# Tạo các input cho người dùng nhập liệu
age = st.number_input("Tuổi", min_value=0, max_value=120, value=25)
sex = st.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
cp = st.selectbox("Loại đau ngực", options=[0, 1, 2, 3])
trestbps = st.number_input("Huyết áp tâm thu khi nghỉ (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Mức độ Cholesterol trong máu (mg/dl)", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Đường huyết lúc đói >120", options=[0, 1], format_func=lambda x: "Không (> 120 mg/dl)" if x == 0 else "Có (<= 120 mg/dl)")
restecg = st.selectbox("Kết quả điện tâm đồ lúc nghỉ", options=[0, 1, 2])
thalach = st.number_input("Nhịp tim tối đa đạt được khi gắng sức", min_value=60, max_value=220, value=150)
exang = st.selectbox("Đau thắt ngực do gắng sức", options=[0, 1])
oldpeak = st.number_input("Sự chênh lệch ST khi gắng sức so với lúc nghỉ ngơi", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Độ dốc của đoạn ST khi gắng sức", options=[0, 1, 2], format_func=lambda x: "Tăng dần" if x == 0 else "Phẳng" if x == 1 else "Giảm dần")
ca = st.selectbox("Số lượng mạch chính bị hẹp", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Tình trạng Thalassemia", options=[0, 1, 2, 3])

if st.button("Dự đoán"):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    if model is not None:
        prediction = predict_heart_disease(model, input_data, selected_features_indices)
        if prediction == 1:
            st.write("Nguy cơ mắc bệnh tim.")
        else:
            st.write("Không có nguy cơ mắc bệnh tim.")
    else:
        st.write("Vui lòng chọn một mô hình hợp lệ.")
