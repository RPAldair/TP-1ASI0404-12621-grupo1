# Mostrar matriz de confusión (normalizada por fila) y guardarla
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# rutas (ajusta si cambiaste)
MODEL_PATH = 'models/nb_model.pkl'
VECT_PATH = 'models/vectorizer.pkl'
TEST_PATH = 'data/test.jsonl'

# cargar
clf = pickle.load(open(MODEL_PATH, 'rb'))
vect = pickle.load(open(VECT_PATH, 'rb'))
test = [json.loads(l) for l in open(TEST_PATH, 'r', encoding='utf-8')]

X_test = [' '.join(t['pictos']) for t in test]
y_test = [t['intent'] for t in test]
y_pred = clf.predict(vect.transform(X_test))

labels = sorted(list(set(y_test)))  # orden de etiquetas (puedes fijar el orden deseado)
cm = confusion_matrix(y_test, y_pred, labels=labels)

# normalizar por fila (recall por clase)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.2f')
plt.title('Matriz de confusión (normalizada por clase verdadera)')
plt.tight_layout()
import os
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/confusion.png', dpi=150)
plt.show()