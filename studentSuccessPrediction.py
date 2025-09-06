from sklearn.tree import DecisionTreeClassifier


X = [
    [2, 1],
    [4, 2],
    [6, 2],
    [8, 3],
]
y = [0, 0, 1, 1]


model = DecisionTreeClassifier()
model.fit(X, y)


newStudent = [[7, 2]]
estimate = model.predict(newStudent)

print("Result:", "passed" if estimate[0] == 1 else "failed")
