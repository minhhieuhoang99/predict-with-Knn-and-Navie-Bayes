import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
# import module
import pandas as pd
# Get the Keys
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key 

def transform_gender(x):
    if x == 'Female':
        return 1
    elif x == 'Male':
        return 0
    else:
        return -1
    
def transform_customer_type(x):
    if x == 'Loyal Customer':
        return 1
    elif x == 'disloyal Customer':
        return 0
    else:
        return -1
    
def transform_travel_type(x):
    if x == 'Business travel':
        return 1
    elif x == 'Personal Travel':
        return 0
    else:
        return -1
    
def transform_class(x):
    if x == 'Business':
        return 2
    elif x == 'Eco Plus':
        return 1
    elif x == 'Eco':
        return 0    
    else:
        return -1
    
def transform_satisfaction(x):
    if x == 'satisfied':
        return 1
    elif x == 'neutral or dissatisfied':
        return 0
    else:
        return -1
    
def process_data(df):
    df = df.drop(['Unnamed: 0', 'id'], axis = 1)
    df['Gender'] = df['Gender'].apply(transform_gender)
    df['Customer Type'] = df['Customer Type'].apply(transform_customer_type)
    df['Type of Travel'] = df['Type of Travel'].apply(transform_travel_type)
    df['Class'] = df['Class'].apply(transform_class)
    df['satisfaction'] = df['satisfaction'].apply(transform_satisfaction)
    df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace = True) #điền giá trị thiếu
    
    return df

# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test,knn_slider):
    knn = KNeighborsClassifier(n_neighbors=knn_slider)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, knn

# Training Naive-bayes
@st.cache(suppress_st_warning=True)
def Naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb_model = gnb.fit(X_train, y_train)
    y_pred = gnb_model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, gnb

def accept_user_data():
    # id = st.text_input("Enter the id number: ")
    Gender = st.selectbox('Giới tính',tuple(Gender_label.keys()))
    Customer_Type = st.selectbox('Khách hàng',tuple(Customer_Type_label.keys()))
    Age = st.slider ("Tuổi: ",7,100)
    Type_of_Travel = st.selectbox('Loại hình chuyến bay',tuple(Type_of_Travel_label.keys()))
    Class = st.selectbox('Hạng vé',tuple(Class_label.keys()))
    Flight_Distance = st.slider("Khoảng cách chuyến bay:",30,5000)
    Inflight_wifi_service   = st.slider("Đánh giá dịch vụ wifi:",0,5)
    Departure_Arrival_time_convenient   = st.slider ("Đánh giá độ thuận tiện của thời gian đi/đến: ",0,5)
    Ease_of_Online_booking  = st.slider ("Đánh giá dịch vụ đặt vé trực tuyến: ",0,5)
    Gate_location   = st.slider ("Đánh giá vị trí cổng bay: ",0,5)
    Food_and_drink  = st.slider ("Đánh giá đồ ăn/uống: ",0,5)
    Online_boarding = st.slider ("Đánh giá dịch vụ trực tuyến: ",0,5)
    Seat_comfort    = st.slider ("Đánh giá ghế ngồi: ",0,5)
    Inflight_entertainment  = st.slider ("Đánh giá dịch vụ giải trí: ",0,5)
    On_board_service    = st.slider ("Đách giá dịch vụ nhận vé trực tuyến: ",0,5)
    Leg_room_service    = st.slider ("Đánh giá chỗ để chân: ",0,5)
    Baggage_handling    = st.slider ("Đánh giá về hành lý xách tay: ",0,5)
    Checkin_service = st.slider ("Đánh giá dịch vụ checkin: ",0,5)
    Inflight_service    = st.slider ("Đánh giá dịch vụ trên chuyến bay: ",0,5)
    Cleanliness = st.slider ("Đánh giá vệ sinh chuyến bay: ",0,5)
    Departure_Delay_in_Minutes  = st.slider ("Thời gian khởi hành muộn: ",0,1200)
    Arrival_Delay_in_Minutes    = st.slider ("Thời gian kết thúc muộn: ",0,1200)
    k_Gender = get_value(Gender,Gender_label)
    k_Customer_Type = get_value(Customer_Type,Customer_Type_label)
    k_Type_of_Travel = get_value(Type_of_Travel,Type_of_Travel_label)
    k_Class = get_value(Class,Class_label)
    user_prediction_data = np.array([Gender,Customer_Type,Age,Type_of_Travel,Class,Flight_Distance,Inflight_wifi_service,Departure_Arrival_time_convenient,Ease_of_Online_booking,Gate_location,Food_and_drink,Online_boarding,Seat_comfort,Inflight_entertainment,On_board_service,Leg_room_service,Baggage_handling,Checkin_service,Inflight_service,Cleanliness,Departure_Delay_in_Minutes,Arrival_Delay_in_Minutes]).reshape(1,-1)
    user_prediction_data_encode = np.array([k_Gender,k_Customer_Type,Age,k_Type_of_Travel,k_Class,Flight_Distance,Inflight_wifi_service,Departure_Arrival_time_convenient,Ease_of_Online_booking,Gate_location,Food_and_drink,Online_boarding,Seat_comfort,Inflight_entertainment,On_board_service,Leg_room_service,Baggage_handling,Checkin_service,Inflight_service,Cleanliness,Departure_Delay_in_Minutes,Arrival_Delay_in_Minutes]).reshape(1,-1)
    
    return user_prediction_data,user_prediction_data_encode

def load_print_data():
    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Hiển thị dữ liệu'):
        st.subheader("Hiển thị 100 dòng dữ liệu---->>>") 
        st.write(df_train.head(100))
    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Hiển thị dữ liệu mã hóa'):
        st.subheader("Hiển thị 100 dòng dữ liệu mã hóa---->>>") 
        st.write(train.head(100)) 
    

# assign dataset names
list_of_names = ['train','test']
# create empty list
dataframes_list = []
# append datasets into teh list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv("./airline-passenger-satisfaction/"+list_of_names[i]+".csv")
    dataframes_list.append(temp_df) 
# @st.cache
# def loadData():
st.set_option('deprecation.showPyplotGlobalUse', False)   
df_train = dataframes_list[0]
df_test = dataframes_list[1]
    

train = process_data(df_train)
test = process_data(df_test)

features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target = ['satisfaction']

X_train = train[features]
y_train = train[target].to_numpy()
X_test = test[features]
y_test = test[target].to_numpy()
le = LabelEncoder()
y_test = le.fit_transform(y_test.flatten())
y_train =le.fit_transform(y_train.flatten())
# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

Gender_label = {'Nữ': 0, 'Nam': 1}
Customer_Type_label = {'Khách hàng thân thiết': 0, 'Khách hàng thường': 1}
Type_of_Travel_label = {'Công tác': 0, 'Cá nhân': 1}
Class_label = {'Thương gia': 0, 'Thường': 1, 'Tiết kiệm': 2}
satisfaction_label = {'Phân vân hoặc không Hài lòng': 0, 'Hài lòng': 1}
# Accepting user data for predicting its Member Type

def main():
           
    # ML Section
    choose_model = st.sidebar.selectbox("Tùy Chọn",["Báo Cáo", "K-Nearest Neighbours" , "Navie-Bayes"])
    if(choose_model == "Báo Cáo"):
        st.title("Đồ thị hóa dữ liệu")
        st.subheader("1. Dữ liệu thu thập") 
        st.write(df_train.head(100))
        st.write('Tổng số bản ghi dữ liệu huấn luyện:',df_train.shape[0])
        st.write('Tổng số bản ghi dữ liệu thử nghiệm:',df_test.shape[0])
        st.write('Tổng số thuộc Tính:',df_train.shape[1]-3)
        st.write(df_train.dtypes)
        st.write(df_train.describe())
        st.write("Kiểm tra dữ liệu thiếu sót")
        # df_train.columns = [c.replace(' ', '_') for c in train.columns]
        total = df_train.isnull().sum().sort_values(ascending=False)
        percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending=False)
        missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        st.table(missing.head())
        # fig = plt.figure(figsize = (8,5))
        sns.boxplot(x = df_train['Arrival Delay in Minutes'], y = None)
        plt.title('Bảng thống kê thời gian khởi hành muộn', fontsize = 12)
        st.pyplot(plt.show())

        st.subheader("2. Kiểm tra sự cân bằng của biến mục tiêu") 

        # fig = plt.figure(figsize = (8,5))
        # train.satisfaction.value_counts(normalize = True).plot(kind='bar', color= ['darkorange','steelblue'], alpha = 0.9, rot=0)
        # plt.title('Độ cân bằng của biến mục tiêu', fontsize = 15)
        # plt.xlabel('Biến Mục Tiêu', fontsize = 15)
        # plt.ylabel('', fontsize = 15)
        # st.pyplot(fig)

        df_target = df_train['satisfaction']
        df_target.value_counts()
        fig = plt.figure(figsize = (8,5))
        sns.countplot(x = df_target,palette='husl')
        plt.text(x = 0.95, y = df_target.value_counts()[1] + 1, s = str(round((df_target.value_counts()[1])*100/len(df_target),2)) + '%')
        plt.text(x = -0.05, y = df_target.value_counts()[0] +1, s = str(round((df_target.value_counts()[0])*100/len(df_target),2)) + '%')
        plt.title('Độ cân bằng của biến mục tiêu', fontsize = 15)
        plt.xlabel('Biến Mục Tiêu', fontsize = 15)
        plt.ylabel('', fontsize = 15)
        st.pyplot(fig)

        st.subheader("3. Biểu đồ thống kê sự hài lòng theo giới tính") 
        
        pd.crosstab(df_train.Gender,df_train.satisfaction).plot(kind="bar",figsize=(15,6))
        plt.title('Sự hài lòng theo giới tính')
        plt.xticks(rotation=0)
        plt.legend(['neutral/dissatisfied', 'Satisfied'])
        plt.ylabel('Frequency')
        st.pyplot(plt.show())
        

        st.subheader("2. Biểu đồ nhiệt") 
        corr = train.corr(method='spearman')
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(150, 1, as_cmap=True)

        # Set up the matplotlib figure
        heatmap, ax = plt.subplots(figsize=(20, 18))

        

        # Draw the heatmap with the mask and correct aspect ratio
        
        sns.heatmap(corr, annot = True, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5)
        st.pyplot(heatmap)

        st.subheader("3. Xây dựng mô hình")
        st.write("Naive bayes")
        st.write("K-Nearest Neighbor")


        

    if(choose_model == "Navie-Bayes"):
        st.title("Dự đoán sự hài lòng của khách hàng với chuyến bay sự dụng thuật toán Navie-Bayes")
        # X_train, X_test, y_train, y_test, le = process_data(df_train)
        score, report, gnb = Naive_bayes(X_train, X_test, y_train, y_test)
        st.text("Độ chính xác của mô hình Naive_bayes là: ")
        st.write(score,"%")
        st.text("Báo cáo mô hình Naive_bayes: ")
        st.write(report)
        lpd = load_print_data() 
        st.write(lpd)
        user_prediction_data , user_prediction_data_encode = accept_user_data()
        st.write(user_prediction_data)

        try:
            if(st.button(" Dự đoán ")):
                # user_prediction_data = accept_user_data()    
                pred = gnb.predict(user_prediction_data_encode)
                st.write("Mã hóa dữ liệu đánh giá: ", user_prediction_data_encode)
                # st.write("The Predicted Class is: ", pred) # Inverse transform to get the original dependent value.                
                final_result = get_key(pred,satisfaction_label)
                st.write("Kết quả dự đoán")
                st.success(final_result) 
                if final_result =='Phân vân hoặc không Hài lòng':
                    st.write("Khách hàng phân vân hoặc không hài lòng về chuyến bay")
                if final_result =='Hài lòng':
                    st.write("Khách hàng hài lòng về chuyến bay")
        except:
            pass

    

    elif(choose_model == "K-Nearest Neighbours"):
        st.title("Dự đoán sự hài lòng của khách hàng với chuyến bay sự dụng thuật toán K-Nearest Neighbours")
        knn_slider  = st.slider("knn:",3, 101, 55, 2)
        if(knn_slider):
            score, report, knn = Knn_Classifier(X_train, X_test, y_train, y_test,knn_slider)
            st.write("Độ chính xác của mô hình K-Nearest Neighbour với k =",knn_slider," là: ")
            st.write(score,"%")
            st.text("Báo cáo mô hình K-Nearest Neighbour: ") 
            st.write(report)
            lpd = load_print_data()
            # knn_slider = st.slider("knn:",3, 101, 55, 2) 
            st.write(lpd)
            user_prediction_data , user_prediction_data_encode = accept_user_data()
            st.write(user_prediction_data)
        
        try:
            if(st.button(" Dự đoán ")):
                # user_prediction_data = accept_user_data()    
                pred = knn.predict(user_prediction_data_encode)
                st.write("Mã hóa dữ liệu đánh giá: ", user_prediction_data_encode)
                # st.write("The Predicted Class is: ", pred) # Inverse transform to get the original dependent value.                
                final_result = get_key(pred,satisfaction_label)
                st.write("Kết quả dự đoán")
                st.success(final_result) 
                if final_result =='Phân vân hoặc không Hài lòng':
                    st.write("Khách hàng phân vân hoặc không hài lòng về chuyến bay")
                if final_result =='Hài lòng':
                    st.write("Khách hàng hài lòng về chuyến bay")
        except:
            pass

if __name__ == "__main__":
    main()