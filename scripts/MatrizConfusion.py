# ejemplo (ejecutar en entorno .venv)
import pickle, json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

clf = pickle.load(open('models/nb_model.pkl','rb'))
vect = pickle.load(open('models/vectorizer.pkl','rb'))

test = [json.loads(l) for l in open('data/test.jsonl','r',encoding='utf-8')]
X_test = [ ' '.join(t['pictos']) for t in test ]
y_test = [ t['intent'] for t in test ]
y_pred = clf.predict(vect.transform(X_test))

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', xticks_rotation=45)
plt.show()