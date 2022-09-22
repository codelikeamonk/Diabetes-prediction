pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

df = pd.read_csv("diabetes.csv")
df.head()

df.shape

df.describe()

X = df.drop("Outcome",axis=1)
y= df["Outcome"] #We will predict Outcome(diabetes) 

X_train = X.iloc[:600]
X_test = X.iloc[600:]
y_train = y[:600]
y_test = y[600:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)

nb = GaussianNB().fit(X_train,y_train)

nb

nb.predict(X_test)[:10]

y_pred = nb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

cm

print("Our Accuracy is: ", (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))

accuracy_score(y_test,y_pred)

recall_score(y_test,y_pred)

precision_score(y_test,y_pred)

f1_score(y_test,y_pred)

print(classification_report(y_test,y_pred))


