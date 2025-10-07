#!/usr/bin/env python3
"""
train_nb.py
Entrena un clasificador Multinomial Naive Bayes para predecir la intención a partir de
secuencias de pictogramas (tokens). Usa CountVectorizer sobre las secuencias 'pictos'.

Salida: models/nb_model.pkl y models/vectorizer.pkl
"""
import json
from pathlib import Path
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def read_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def pictos_to_text(pictos):
    # unir tokens con espacio para vectorizer
    return ' '.join(pictos)


def load_data():
    train = read_jsonl(DATA_DIR / 'train.jsonl')
    val = read_jsonl(DATA_DIR / 'val.jsonl')
    test = read_jsonl(DATA_DIR / 'test.jsonl')
    return train, val, test


def prepare(items):
    X = [pictos_to_text(it['pictos']) for it in items]
    y = [it['intent'] for it in items]
    return X, y


def main():
    train, val, test = load_data()
    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)
    X_test, y_test = prepare(test)

    vect = CountVectorizer()
    X_train_t = vect.fit_transform(X_train)
    X_val_t = vect.transform(X_val)
    X_test_t = vect.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_t, y_train)

    y_pred = clf.predict(X_val_t)
    print('Validation results:')
    print(classification_report(y_val, y_pred))

    # test
    y_test_pred = clf.predict(X_test_t)
    print('Test results:')
    print(classification_report(y_test, y_test_pred))

    # guardar modelos
    with open(MODELS_DIR / 'nb_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / 'vectorizer.pkl', 'wb') as f:
        pickle.dump(vect, f)
    print('Modelos guardados en', MODELS_DIR)


if __name__ == '__main__':
    main()
