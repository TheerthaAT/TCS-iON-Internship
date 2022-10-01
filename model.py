
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
train_data=pd.read_csv(r"C:\Users\User\Desktop\MobileDataSets/MobileTrain.csv")

#data=train_data[["battery_power","fc","pc","int_memory","mobile_wt","sc_h","sc_w","ram","price_range"]]
data0=train_data[["px_height","px_width","battery_power","fc","pc","int_memory","mobile_wt","sc_h","sc_w","ram","price_range"]]

X = data0.drop("price_range" , axis=1)
y = data0["price_range"]

from sklearn.preprocessing import StandardScaler


scaler= StandardScaler()
X=scaler.fit_transform(X)
X=pd.DataFrame(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression()
logit_model.fit(X_train,y_train)



pickle.dump(logit_model,open('model.pkl','wb'))
