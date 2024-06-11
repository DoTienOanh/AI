# Thêm thư viện
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from scipy.stats import uniform
st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("***BÀI TOÁN DỰ ĐOÁN VỀ BỆNH TIM***")
st.write("**I. Phát biểu bài toán**")
st.write("Bài toán đặt ra là xây dựng một mô hình dự đoán khả năng mắc bệnh tim dựa trên các thông tin y tế và lối sống của bệnh nhân. Mục tiêu là xác định liệu một người có nguy cơ cao mắc bệnh tim hay không, từ đó hỗ trợ trong việc chẩn đoán và điều trị sớm.")

st.write("**II. Giới thiệu về bộ dữ liệu**")
data = pd.read_csv('streamlit/heart.csv')
st.write('Kích thước bộ dữ liệu đang tìm hiểu là', data.shape)

st.write("+ Lấy một số dữ liệu đầu")
st.table(data.head())

st.write("+ Kiểu dữ liệu của mỗi cột")
st.table(data.dtypes)

# +

data = data[data['ca'] < 4] # xóa các giá trị sai ở cột 'ca'
data = data[data['thal'] > 0] # xóa các giá trị sai ở cột 'thal'
print(f'Độ dài của dữ liệu bây giờ là {len(data)} thay vì 303')
# -

st.write("**III. Xây dựng bài toán**")
st.write("*3.1. Tiền xử lý dữ liệu*")
st.write("Nguồn dữ liệu mà chúng tôi sẽ sử dụng lấy từ một trong các bộ dữ liệu phổ biến về bệnh tim trên Kaggle. Đó là bộ dữ liệu Heart Disease UCI .")
st.write("Bộ dữ liệu này thường bao gồm các thông tin liên quan đến bệnh nhân như tuổi, giới tính, mức cholesterol, huyết áp, nhịp tim, và các thông số y tế khác.")

st.write("*3.2. Tiền xử lý dữ liệu*")
st.write("+ Nhóm đã thay đổi tên của các thuộc tính để có thể dễ dàng làm việc với database:")
data = data.rename(
    columns = {'age': 'Tuổi',
               'sex': 'Giới tính',
               'cp':'Loại đau ngực', 
               'trestbps':'Huyết áp khi nghỉ ngơi', 
               'chol': 'Cholesterol trong máu',
               'fbs': 'Đường huyết lúc đói',
               'restecg' : 'Kết quả điện tâm đồ lúc nghỉ ngơi', 
               'thalach': 'Nhịp tim tối đa khi gắng sức', 
               'exang': 'Đau thắt ngực do gắng sức',
               'oldpeak': 'Sự chênh lệch ST', 
               'slope': 'Độ dốc đoạn ST', 
               'ca':'Số mạch máu chính bị hẹp', 
               'thal': 'Thalassemia'}, 
    errors="raise")
# +
data['Giới tính'][data['Giới tính'] == 0] = 'Nữ'
data['Giới tính'][data['Giới tính'] == 1] = 'Nam'

data['Loại đau ngực'][data['Loại đau ngực'] == 0] = 'Đau thắt điển hình'
data['Loại đau ngực'][data['Loại đau ngực'] == 1] = 'Đau thắt ngực không điển hình'
data['Loại đau ngực'][data['Loại đau ngực'] == 2] = 'Không đau thắt'
data['Loại đau ngực'][data['Loại đau ngực'] == 3] = 'Không có triệu chứng'

data['Đường huyết lúc đói'][data['Đường huyết lúc đói'] == 0] = 'Thấp hơn 120mg/ml'
data['Đường huyết lúc đói'][data['Đường huyết lúc đói'] == 1] = 'Cao hơn 120mg/ml'

data['Kết quả điện tâm đồ lúc nghỉ ngơi'][data['Kết quả điện tâm đồ lúc nghỉ ngơi'] == 0] = 'Bình thường'
data['Kết quả điện tâm đồ lúc nghỉ ngơi'][data['Kết quả điện tâm đồ lúc nghỉ ngơi'] == 1] = 'ST-T bất thường'
data['Kết quả điện tâm đồ lúc nghỉ ngơi'][data['Kết quả điện tâm đồ lúc nghỉ ngơi'] == 2] = 'Bất thường khác'

data['Đau thắt ngực do gắng sức'][data['Đau thắt ngực do gắng sức'] == 0] = 'Không'
data['Đau thắt ngực do gắng sức'][data['Đau thắt ngực do gắng sức'] == 1] = 'Có'

data['Độ dốc đoạn ST'][data['Độ dốc đoạn ST'] == 0] = 'Dốc lên'
data['Độ dốc đoạn ST'][data['Độ dốc đoạn ST'] == 1] = 'Bằng phẳng'
data['Độ dốc đoạn ST'][data['Độ dốc đoạn ST'] == 2] = 'Dốc xuống'

data['Thalassemia'][data['Thalassemia'] == 1] = 'Khuyết tật cố định'
data['Thalassemia'][data['Thalassemia'] == 2] = 'Bình thường'
data['Thalassemia'][data['Thalassemia'] == 3] = 'Khuyết tật có thể chỉnh'
# -
st.table(data.dtypes)

st.table(data.head())

st.write("*2.2. Trích chọn đặc trưng*")
st.write("+ Visualize data tìm mối tương quan giữa các thuộc tính và mức ảnh hưởng")

# Thuộc tính số
num_feats = ['Tuổi', 'Cholesterol trong máu', 'Huyết áp khi nghỉ ngơi', 'Nhịp tim tối đa khi gắng sức', 'Sự chênh lệch ST', 'Số mạch máu chính bị hẹp']
# Phân loại (binary)
bin_feats = ['Giới tính', 'Đường huyết lúc đói', 'Đau thắt ngực do gắng sức', 'target']
# Phân loại (multi-)
nom_feats= ['Loại đau ngực', 'Kết quả điện tâm đồ lúc nghỉ ngơi', 'Độ dốc đoạn ST', 'Thalassemia']
cat_feats = nom_feats + bin_feats

# +
mypal= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']

plt.figure(figsize=(7, 5),facecolor='White')
total = float(len(data))
ax = sns.countplot(x=data['target'], palette=mypal[1::4])
ax.set_facecolor('white')

for p in ax.patches:
    
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.1f} %'.format((height/total)*100), ha="center",
           bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

ax.set_title('Phân phối biến mục tiêu', fontsize=20, y=1.05)
sns.despine(right=True)
sns.despine(offset=5, trim=True)
st.pyplot()
# -

st.write("+ Thống kê tóm tắt cho từng cột số:")
# .T: để chuyển từ cột sang hàng
data[num_feats].describe().T

# +
st.write("+ Các biểu đồ phân tán:")
import seaborn as sns
import matplotlib.pyplot as plt

# Giả sử data là DataFrame của bạn
data['target'] = data['target'].astype(str)



# +
L = len(num_feats)
ncol= 2
nrow= int(np.ceil(L/ncol))
#remove_last= (nrow * ncol) - L

fig, ax = plt.subplots(nrow, ncol, figsize=(16, 14),facecolor='#F6F5F4')   
fig.subplots_adjust(top=0.92)

i = 1
for col in num_feats:
    plt.subplot(nrow, ncol, i, facecolor='#F6F5F4')
    
    ax = sns.kdeplot(data=data, x=col, hue="target", multiple="stack", palette=mypal[1::4]) 
    ax.set_xlabel(col, fontsize=20)
    ax.set_ylabel("Mật độ", fontsize=20)
    sns.despine(right=True)
    sns.despine(offset=0, trim=False)
    
    if col == 'Số mạch máu chính bị hẹp':
        sns.countplot(data=data, x=col, hue="target", palette=mypal[1::4])
        for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format((height)),ha="center",
                      bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
    
    i = i +1
plt.suptitle('Phân phối các thuộc tính số' ,fontsize = 24);
st.pyplot()

# +

a = ['Tuổi', 'Cholesterol trong máu', 'Kết quả điện tâm đồ lúc nghỉ ngơi', 'Nhịp tim tối đa khi gắng sức', 'Sự chênh lệch ST', 'target']
data_ = data[a]
g = sns.pairplot(data_, hue="target", corner=True, diag_kind='hist', palette=mypal[1::4]);
plt.suptitle('Cặp đồ thị: Thuộc tính số ' ,fontsize = 24);
st.pyplot()


# +
# fig, ax = plt.subplots(1,4, figsize=(22, 4))
# sns.regplot(data=data[data['target'] ==1], x='Tuổi', y='Cholesterol trong máu', ax = ax[0], color=mypal[0], label='1')
# sns.regplot(data=data[data['target'] ==0], x='Tuổi', y='Cholesterol trong máu', ax = ax[0], color=mypal[5], label='0')
# sns.regplot(data=data[data['target'] ==1], x='Tuổi', y='Nhịp tim tối đa khi gắng sức', ax = ax[1], color=mypal[0], label='1')
# sns.regplot(data=data[data['target'] ==0], x='Tuổi', y='Nhịp tim tối đa khi gắng sức', ax = ax[1], color=mypal[5], label='0')
# sns.regplot(data=data[data['target'] ==1], x='Tuổi', y='Kết quả điện tâm đồ lúc nghỉ ngơi', ax = ax[2], color=mypal[0], label='1')
# sns.regplot(data=data[data['target'] ==0], x='Tuổi', y='Kết quả điện tâm đồ lúc nghỉ ngơi', ax = ax[2], color=mypal[5], label='0')
# sns.regplot(data=data[data['target'] ==1], x='Tuổi', y='Sự chênh lệch ST', ax = ax[3], color=mypal[0], label='1')
# sns.regplot(data=data[data['target'] ==0], x='Tuổi', y='Sự chênh lệch ST', ax = ax[3], color=mypal[5], label='0')
# plt.suptitle('Biểu đồ của các thuộc tính được chọn')
# plt.legend();

# +
def count_plot(data, cat_feats):    
    L = len(cat_feats)
    ncol= 2
    nrow= int(np.ceil(L/ncol))
    remove_last= (nrow * ncol) - L

    fig, ax = plt.subplots(nrow, ncol,figsize=(18, 24), facecolor='#F6F5F4')    
    fig.subplots_adjust(top=0.92)
    ax.flat[-remove_last].set_visible(False)

    i = 1
    for col in cat_feats:
        plt.subplot(nrow, ncol, i, facecolor='white')
        ax = sns.countplot(data=data, x=col, hue="target", palette=mypal[1::4])
        ax.set_xlabel(col, fontsize=20)
        ax.set_ylabel("Số lượng", fontsize=20)
        sns.despine(right=True)
        sns.despine(offset=0, trim=False) 
        plt.legend(facecolor='white')
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format((height)),ha="center",
                  bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
        
        i = i +1

    plt.suptitle('Phân phối các thuộc tính phân loại' ,fontsize = 24)
    return 0

count_plot(data, cat_feats[0:-1]);
st.pyplot()
# -

st.write("+ Các biểu đồ tương quan:")
df_ = data[num_feats]
corr = df_.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
cmap = sns.color_palette(mypal, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, annot=True,
            square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
ax.set_title("Tương quan thuộc tính số", fontsize=20, y= 1.05);
st.pyplot()

# +
feats_ = ['Tuổi', 'Cholesterol trong máu', 'Huyết áp khi nghỉ ngơi', 'Nhịp tim tối đa khi gắng sức', 'Sự chênh lệch ST', 'Số mạch máu chính bị hẹp', 'target']

def point_biserial(x, y):
    pb = stats.pointbiserialr(x, y)
    return pb[0]
# Chuyển đổi các cột cần thiết sang kiểu số nếu có thể
for col in feats_:
    data[col] = pd.to_numeric(data[col], errors='coerce')
rows= []
for x in feats_:
    col = []
    for y in feats_ :
        pbs =point_biserial(data[x], data[y]) 
        col.append(round(pbs,2))  
    rows.append(col)  
    
pbs_results = np.array(rows)
DF = pd.DataFrame(pbs_results, columns = data[feats_].columns, index =data[feats_].columns)

mask = np.triu(np.ones_like(DF, dtype=bool))
corr = DF.mask(mask)

f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
cmap = sns.color_palette(mypal, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1, center=0, annot=True,
            square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
ax.set_title("Tương quan thuộc tính số so với target", fontsize=20, y= 1.05);
st.pyplot()


# +
# the cramers_v function is copied from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

def cramers_v(x, y): 
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# calculate the correlation coefficients using the above function
data_ = data[cat_feats]
rows= []
for x in data_:
    col = []
    for y in data_ :
        cramers =cramers_v(data_[x], data_[y]) 
        col.append(round(cramers,2))
    rows.append(col)
    
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = data_.columns, index = data_.columns)

# color palette 
mypal_1= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA', '#FC05FB', '#FEAEFE', '#FCD2FC']
# plot the heat map
mask = np.triu(np.ones_like(df, dtype=bool))
corr = df.mask(mask)
f, ax = plt.subplots(figsize=(10, 6), facecolor=None)
cmap = sns.color_palette(mypal_1, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0, annot=True,
            square=False, linewidths=.01, cbar_kws={"shrink": 0.75})
ax.set_title("Tương quan các thuộc tính phân loại", fontsize=20, y= 1.05);
st.pyplot()

st.write("+ Trích chọn đặc trưng:") 
features = ['Tuổi','Giới tính','Huyết áp khi nghỉ ngơi','Cholesterol trong máu','Kết quả điện tâm đồ lúc nghỉ ngơi','Nhịp tim tối đa khi gắng sức',
           'Đau thắt ngực do gắng sức','Sự chênh lệch ST','Độ dốc đoạn ST','Số mạch máu chính bị hẹp','Thalassemia']
target = ['target']
X = data[features]
y = data[target]
st.table(X.head())
st.table(y.head())

# +
st.write("3.4. Xây dựng mô hình")
from sklearn.preprocessing import LabelEncoder

def lb_cat_feats(data, cat_feats):
    lb = LabelEncoder()
    data_encoded = data.copy()
    
    for col in cat_feats:
        data_encoded[col] = lb.fit_transform(data[col])
    
    data = data_encoded
    
    return data


# +
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_curve, auc

def score_summmary(names, classifiers):
    cols=["Classifier", "Accurancy", "ROC_AUC", "Recall", "Precision", "F1"]
    data_table=pd.DataFrame(columns=cols)

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        pred=clf.predict(X_test)
        accurancy=accuracy_score(y_test,pred)

        pred_proba = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, pred_proba)        
        roc_auc = auc(fpr, tpr)

        # confusion matric, cm
        cm = confusion_matrix(y_test, pred) 

        # recall: TP/(TP+FN)
        recall = cm[1,1]/(cm[1,1] +cm[1,0])

        # precision: TP/(TP+FP)
        precision = cm[1,1]/(cm[1,1] +cm[0,1])

        # F1 score: TP/(TP+FP)
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, accurancy*100, roc_auc, recall, precision, f1]], columns=cols)
        data_table = pd.concat([data_table, df], ignore_index=True)     

    return(np.round(data_table.reset_index(drop=True), 2))



# +
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Split datasets
data = lb_cat_feats(data,cat_feats)
features = data.columns[:-1]

X = data[features]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

names=["LogisticsRegression", "NuSVC", "DecisionTree"]
classifiers=[
    LogisticRegression(solver="liblinear", random_state=0),
    NuSVC(probability=True, random_state=0),
    DecisionTreeClassifier(random_state=0),
]
# -

score_summmary(names, classifiers).sort_values(by='Accurancy' , ascending = False)\
.style.background_gradient(cmap='coolwarm')\
.bar(subset=["ROC_AUC",], color='#6495ED')\
.bar(subset=["Recall"], color='#ff355d')\
.bar(subset=["Precision"], color='lightseagreen')\
.bar(subset=["F1"], color='gold')

# Lựa chọn thuật toán LogisticRegression để tiếp tục Điều chỉnh tham số

# +
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.write("+ Các Confusion matrix của 3 thuật toán:")

logistic_model = LogisticRegression(random_state=0)
decision_tree_model = DecisionTreeClassifier(random_state=0)
nusvc_model = NuSVC(random_state=0)

logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
nusvc_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)
nusvc_predictions = nusvc_model.predict(X_test)

logistic_cm = confusion_matrix(y_test, logistic_predictions, labels=logistic_model.classes_)
decision_tree_cm = confusion_matrix(y_test, decision_tree_predictions, labels=decision_tree_model.classes_)
nusvc_cm = confusion_matrix(y_test, nusvc_predictions, labels=nusvc_model.classes_)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, cm, model_name in zip(axes, [logistic_cm, decision_tree_cm, nusvc_cm], ['LogisticRegression', 'DecisionTree', 'NuSVC']):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")

st.pyplot()

# +
#Parameter Tuning: Điều chỉnh tham số cho thuật toán Logistic Regression bằng Randomized Search
#Có điều chỉnh lại pp chia dataset bằng Repeated Stratufied KFold

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import RepeatedStratifiedKFold


lr = LogisticRegression(tol=1e-4, max_iter=1000, random_state=0)

space = dict(C=uniform(loc=0, scale=5),
                     penalty=['l2', 'l1'],
                     solver= ['liblinear'])

search = RandomizedSearchCV(lr, 
                         space, 
                         random_state=0,
                         cv = 5, 
                         scoring='f1')

rand_search = search.fit(X_train, y_train)

print('Best Hyperparameters: %s' % rand_search.best_params_)

# +
from sklearn.metrics import classification_report

params = rand_search.best_params_
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
print(classification_report(y_test, lr.predict(X_test)))

# +
# Tạo ma trận nhầm lẫn (confusion matrix)
cm = confusion_matrix(y_test, lr.predict(X_test))

# Hiển thị ma trận nhầm lẫn
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
st.pyplot()
# -

# Nhận xét: Việc điều chỉnh tham số ko làm tăng hiệu suất của mô hình chứng tỏ ngay từ đầu mô hình Logistic Regression đã cho kết quả đáng tin cậy

# Ngoài các thuật toán phân lớp truyền thống trên, nhóm đề xuất thực hiện thêm 1 thuật toán ML là LGBM để xem mô hình học máy có thể cải thiện hiệu suất hơn nữa không

# +
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Khởi tạo mô hình LightGBM với tham số mặc định
lgbm = lgb.LGBMClassifier()

# Huấn luyện mô hình với callbacks
lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         callbacks=[lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(50)])

# Đánh giá mô hình
print(classification_report(y_test, lgbm.predict(X_test)))

# +
# Tạo ma trận nhầm lẫn (confusion matrix)
cm = confusion_matrix(y_test, lgbm.predict(X_test))

# Hiển thị ma trận nhầm lẫn
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
# -

# Với thuật toán LGBM, ta có thể làm tăng độ đo Recall lên 89%, đồng thời giảm số lượng FN xuống 4 (so với 5 khi chạy mô hình Logistic Regression). Tuy nhiên lại làm giảm số lượng TP từ 32 xuống 31. 
st.pyplot()