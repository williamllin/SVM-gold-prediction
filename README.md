# SVM-gold-prediction
 
# [1] Import Required Libraries
 
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

 
# [2] Define Tickers and Download Data
 
tickers = {
    # Features (X)
    'PALL': 'PALL',
    'SLV': 'SLV',
    'PLG': 'PLG',
    'ALI=F': 'ALI=F',
    'USO': 'USO',
    'CL=F': 'CO',
    'BZ=F': 'BCO',
    'EURUSD=X': 'EUR/USD',
    'DX-Y.NYB': 'USD',
    '^DJI': 'DJ',
    '^GSPC': 'S&P 500',
    'ILTB': 'USD Bond',
    'AGZ': 'AGZ',
    'TIP': 'TIP',
    'DJP': 'DJP',

    # Targets (Y)
    'IAU':'iShares',
    'SGOL':'SGOL',
    'PHYS':'Sprott'
}

data_frames = []
for ticker, prefix in tickers.items():
    data = yf.download([ticker], period='10y')
    data.columns = [f'{prefix}_{col}' for col in data.columns]
    data_frames.append(data)

combined_df = pd.concat(data_frames, axis=1)

 
# [3] Extract Today and Yesterday's Data
 
today_df = combined_df.tail(1).copy()
combined_df = combined_df.iloc[:-1]
yesterday_df = combined_df.tail(1).copy()

 
# [4] Feature Engineering - X Construction
 
adj_close_columns = [col for col in combined_df.columns if col.endswith('_Adj Close')]
low_columns = [col for col in combined_df.columns if col.endswith('_Low')]
Open_columns = [col for col in combined_df.columns if col.endswith('_Open')]

adj_close_df = combined_df[adj_close_columns]
low_df = combined_df[low_columns]
Open_df = combined_df[Open_columns]

low_df_shifted = low_df.shift(1).reindex(Open_df.index)
result_values = Open_df.values - low_df_shifted.values
new_columns = [col.replace('_Open', '_increased') for col in Open_df.columns]
increased_df = pd.DataFrame(result_values, columns=new_columns, index=Open_df.index)
dropped_X_df = increased_df.dropna()
X_df = dropped_X_df.drop(columns=['iShares_increased', 'SGOL_increased', 'Sprott_increased'])

 
# [5] Target Construction - Y Classification
 
adj_close_df_shifted = adj_close_df.shift(1).reindex(Open_df.index)
diff_df = adj_close_df.diff()
dropped_Y_df = diff_df.dropna()

Y_df = pd.DataFrame([
    dropped_Y_df['iShares_Adj Close'],
    dropped_Y_df['SGOL_Adj Close'],
    dropped_Y_df['Sprott_Adj Close']
]).T

Y_df.columns = ["iShares_increased", "SGOL_increased", "Sprott_increased"]

def classify_value_y(y):
    return "Up" if y > 0 else "Down"

y_iShares = Y_df.filter(like='iShares').applymap(classify_value_y)
y_SGOL = Y_df.filter(like='SGOL').applymap(classify_value_y)
y_Sprott = Y_df.filter(like='Sprott').applymap(classify_value_y)

 
# [6] Train/Test Split and Standardization
 
X_train_df, X_test_df, y_iShares_train, y_iShares_test = train_test_split(X_df, y_iShares, test_size=0.2, random_state=77)
_, _, y_SGOL_train, y_SGOL_test = train_test_split(X_df, y_SGOL, test_size=0.2, random_state=77)
_, _, y_Sprott_train, y_Sprott_test = train_test_split(X_df, y_Sprott, test_size=0.2, random_state=77)

scaler = StandardScaler()
X_train_standardized_df = pd.DataFrame(scaler.fit_transform(X_train_df), index=X_train_df.index, columns=X_train_df.columns)
X_test_standardized_df = pd.DataFrame(scaler.transform(X_test_df), index=X_test_df.index, columns=X_test_df.columns)

 
# [7] Categorize Features into Levels
 
def classify_value_X(x):
    if x > 3: return "Very High"
    elif x > 2: return "High"
    elif x > 1: return "Medium High"
    elif x > 0: return "Slightly High"
    elif x < -3: return "Very Low"
    elif x < -2: return "Low"
    elif x < -1: return "Medium Low"
    else: return "Slightly Low"

X_train_classified_df = X_train_standardized_df.applymap(classify_value_X)
X_test_classified_df = X_test_standardized_df.applymap(classify_value_X)

 
# [8] One-Hot Encode Features
 
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_train_classified_df)

X_train_encoded_df = pd.DataFrame(ohe.transform(X_train_classified_df), index=X_train_classified_df.index, columns=ohe.get_feature_names_out())
X_test_encoded_df = pd.DataFrame(ohe.transform(X_test_classified_df), index=X_test_classified_df.index, columns=ohe.get_feature_names_out())

 
# [9] Label Encode Targets
 
le = LabelEncoder()
y_iShares_train_encoded = le.fit_transform(y_iShares_train.values.ravel())
y_iShares_test_encoded = le.transform(y_iShares_test.values.ravel())

y_SGOL_train_encoded = le.fit_transform(y_SGOL_train.values.ravel())
y_SGOL_test_encoded = le.transform(y_SGOL_test.values.ravel())

y_Sprott_train_encoded = le.fit_transform(y_Sprott_train.values.ravel())
y_Sprott_test_encoded = le.transform(y_Sprott_test.values.ravel())

 
# [10] Train and Tune SVM Classifiers
 
param_grid = {
    'C': [0.1, 0.5, 0.8, 1, 10, 100],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(LinearSVC(dual=False), param_grid, cv=5, scoring='accuracy')

#iShares
grid_search.fit(X_train_encoded_df, y_iShares_train_encoded)
best_model_iShares = LinearSVC(dual=False, **grid_search.best_params_)
best_model_iShares.fit(X_train_encoded_df, y_iShares_train_encoded)
y_iShares_pred = best_model_iShares.predict(X_test_encoded_df)

#SGOL
grid_search.fit(X_train_encoded_df, y_SGOL_train_encoded)
best_model_SGOL = LinearSVC(dual=False, **grid_search.best_params_)
best_model_SGOL.fit(X_train_encoded_df, y_SGOL_train_encoded)
y_SGOL_pred = best_model_SGOL.predict(X_test_encoded_df)

#Sprott
grid_search.fit(X_train_encoded_df, y_Sprott_train_encoded)
best_model_Sprott = LinearSVC(dual=False, **grid_search.best_params_)
best_model_Sprott.fit(X_train_encoded_df, y_Sprott_train_encoded)
y_Sprott_pred = best_model_Sprott.predict(X_test_encoded_df)

 
# [11] Evaluation Results
 
print("Best parameters for iShares:", grid_search.best_params_)
print("iShares Misclassified samples:", (y_iShares_test_encoded != y_iShares_pred).sum())
print("iShares Test accuracy:", accuracy_score(y_iShares_test_encoded, y_iShares_pred))
print("iShares F1 score:", f1_score(y_iShares_test_encoded, y_iShares_pred, average="weighted"))

print("Best parameters for SGOL:", grid_search.best_params_)
print("SGOL Misclassified samples:", (y_SGOL_test_encoded != y_SGOL_pred).sum())
print("SGOL Test accuracy:", accuracy_score(y_SGOL_test_encoded, y_SGOL_pred))
print("SGOL F1 score:", f1_score(y_SGOL_test_encoded, y_SGOL_pred, average="weighted"))

print("Best parameters for Sprott:", grid_search.best_params_)
print("Sprott Misclassified samples:", (y_Sprott_test_encoded != y_Sprott_pred).sum())
print("Sprott Test accuracy:", accuracy_score(y_Sprott_test_encoded, y_Sprott_pred))
print("Sprott F1 score:", f1_score(y_Sprott_test_encoded, y_Sprott_pred, average="weighted"))

 
# [12] Predict for Latest Day
 
today_open = today_df.filter(like='_Open')
yesterday_low = yesterday_df.filter(like='_Low')

latest_result_values = today_open.values - yesterday_low.values
latest_X_df = pd.DataFrame(latest_result_values, columns=new_columns, index=today_df.index)
latest_X_df = latest_X_df.drop(columns=['iShares_increased', 'SGOL_increased', 'Sprott_increased'])

latest_X_standardized_df = pd.DataFrame(scaler.transform(latest_X_df), index=latest_X_df.index, columns=latest_X_df.columns)
latest_X＿classified_df = latest_X_standardized_df.applymap(classify_value_X)

latest_X_encoded_df = pd.DataFrame(ohe.transform(latest_X＿classified_df), index=latest_X_df.index, columns=ohe.get_feature_names_out())

latest_iShares_pred = best_model_iShares.predict(latest_X_encoded_df)
latest_SGOL_pred = best_model_SGOL.predict(latest_X_encoded_df)
latest_Sprott_pred = best_model_Sprott.predict(latest_X_encoded_df)

latest_predictions = {
    "iShares": le.inverse_transform(latest_iShares_pred),
    "SGOL": le.inverse_transform(latest_SGOL_pred),
    "Sprott": le.inverse_transform(latest_Sprott_pred)
}

print("\nPrediction for the latest day:")
print("iShares:", latest_predictions["iShares"][0])
print("SGOL:", latest_predictions["SGOL"][0])
print("Sprott:", latest_predictions["Sprott"][0])
