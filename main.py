import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Подключение классификатора дерева решений
from sklearn.model_selection import train_test_split # Подключение функции для разделения выбьорки для обучения и теста
from sklearn import metrics # Подключение метрик

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

#Ставим https://graphviz.org/download/
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'


# Считываем дата сет
candidate = pd.read_csv("data/data.csv")

print(candidate)

n = {'En': 0, 'Fr': 1, 'Ch': 2}
candidate['Language'] = candidate['Language'].map(n)

t = {'MD': 0, 'BD': 1, 'GSE': 2}
candidate['Education'] = candidate['Education'].map(t)


print(candidate)

# Разбиваем дата сет на признаки и результат
feature_cols = ['Language', 'Education', 'Experience', 'NativeSpeaker', 'NoBadHabits']
X = candidate[feature_cols] # Features
y = candidate.Result # Результирующий столбец

# Разбиваем дата сет
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% обучение и 20% тест

# Создаем классификатор дерева решения
clf = DecisionTreeClassifier()

# Тренируем дерево решения
clf = clf.fit(X_train,y_train)

# Предсказываем и тестируем на результат (сравнивая то что дает дерево с 30% сетом)
y_pred = clf.predict(X_test)

# Выводим отчет, на сколько наше дерево точно
print("Точность:",metrics.accuracy_score(y_test, y_pred))

# Получаем картинку
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('candidate.png')
Image(graph.create_png())

# Предсказание Возьмем ли мы на работу данного кандидата
row = pd.DataFrame([[2, 0, 4, 0,0]],columns=['Language', 'Education', 'Experience', 'NativeSpeaker', 'NoBadHabits'],dtype=float)
print(clf.predict(row))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
