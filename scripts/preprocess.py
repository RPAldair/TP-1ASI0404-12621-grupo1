#!/usr/bin/env python3
"""
preprocess.py
Lee `pictogramas_dataset/dataset_pictogramas.csv`, normaliza palabras y genera un dataset sintético
de pares (secuencia de pictogramas -> texto, intent) en formato JSONL para entrenamiento.

Salida: data/train.jsonl, data/val.jsonl, data/test.jsonl
"""
import os
import json
import random
from pathlib import Path

import pandas as pd
from unidecode import unidecode


ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / 'data' / 'dataset_pictogramas.csv'
OUT_DIR = ROOT / 'data'
OUT_DIR.mkdir(exist_ok=True)


def load_pictos(csv_path):
    df = pd.read_csv(csv_path)
    # Normalizar columna palabra
    df['palabra'] = df['palabra'].astype(str).str.lower().apply(unidecode)
    return df


TEMPLATES = {
    'comida': [
        'Quiero {item}',
        'Me gustaria {item}',
        'Dame {item}, por favor',
        '¿Tienes {item}?'
    ],
    'accion': [
        'Necesito {accion}',
        'Quiero {accion}',
        '¿Puedes {accion}?'
    ],
    'saludo': [
        'Hola',
        'Buenos dias',
        'Buenas tardes',
        'Chau'
    ],
    'lugar': [
        '¿Donde esta {lugar}?',
        'Voy a {lugar}',
        'Estoy en {lugar}'
    ],
    'objeto': [
        'Necesito {obj}',
        'Donde esta mi {obj}?',
        'Tengo un {obj}'
    ]
}


def sample_words_by_category(df):
    cat_map = {}
    for cat, g in df.groupby('categoria'):
        words = g['palabra'].dropna().unique().tolist()
        cat_map[cat] = words
    return cat_map


def word_to_picto(df):
    # mapa palabra -> archivo (nombre de imagen)
    mapping = dict(zip(df['palabra'], df['archivo']))
    return mapping


def generate_examples(df, n_per_intent=400, seed=42):
    random.seed(seed)
    cat_map = sample_words_by_category(df)
    mapping = word_to_picto(df)

    examples = []

    # comida intent: sample from category 'comida' or similar
    food_words = cat_map.get('comida', [])
    acciones = cat_map.get('acciones', [])
    lugares = cat_map.get('lugares', [])
    objetos = cat_map.get('objetos', [])

    for _ in range(n_per_intent):
        # comida
        if food_words:
            item = random.choice(food_words)
            text = random.choice(TEMPLATES['comida']).format(item=item)
            pictos = [item]
            examples.append({'pictos': pictos, 'text': text, 'intent': 'comida'})

        # accion
        if acciones:
            accion = random.choice(acciones)
            text = random.choice(TEMPLATES['accion']).format(accion=accion)
            pictos = [accion]
            examples.append({'pictos': pictos, 'text': text, 'intent': 'accion'})

        # saludo (no pictos necessary but pick common)
        text = random.choice(TEMPLATES['saludo'])
        pictos = [text.lower()] if text else []
        examples.append({'pictos': pictos, 'text': text, 'intent': 'saludo'})

        # lugar
        if lugares:
            lugar = random.choice(lugares)
            text = random.choice(TEMPLATES['lugar']).format(lugar=lugar)
            pictos = [lugar]
            examples.append({'pictos': pictos, 'text': text, 'intent': 'lugar'})

        # objeto
        if objetos:
            obj = random.choice(objetos)
            text = random.choice(TEMPLATES['objeto']).format(obj=obj)
            pictos = [obj]
            examples.append({'pictos': pictos, 'text': text, 'intent': 'objeto'})

    # postprocess: ensure pictos mapping exists; keep only examples with mapped pictos
    filtered = []
    for ex in examples:
        picto_files = []
        for w in ex['pictos']:
            key = unidecode(str(w).lower())
            if key in mapping:
                picto_files.append(key)
            else:
                # keep the word token anyway; vectorizer will use tokens
                picto_files.append(key)
        ex['pictos_norm'] = picto_files
        filtered.append(ex)

    return filtered


def save_jsonl(items, path):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def split_and_save(examples, out_dir=OUT_DIR, seed=42):
    random.seed(seed)
    random.shuffle(examples)
    n = len(examples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    save_jsonl(train, out_dir / 'train.jsonl')
    save_jsonl(val, out_dir / 'val.jsonl')
    save_jsonl(test, out_dir / 'test.jsonl')
    print(f'Saved: train={len(train)}, val={len(val)}, test={len(test)}')


def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f'{DATASET_CSV} not found. Asegurate de tener el csv en pictogramas_dataset/')
    df = load_pictos(DATASET_CSV)
    examples = generate_examples(df, n_per_intent=300)
    # simplify objects to only pictos_norm and intent and text
    items = [{'pictos': e['pictos_norm'], 'text': e['text'], 'intent': e['intent']} for e in examples]
    split_and_save(items)


if __name__ == '__main__':
    main()
