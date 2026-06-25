# Guía de ejecución

## Crear entorno virtual en Ubuntu

```bash
sudo apt install python3.12-venv python3-full -y
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Entrenar modelos

```bash
python train_maternia_integral.py
```

## Ejecutar app

```bash
python -m streamlit run app_streamlit_maternia.py
```

## Probar CSV CTG

Usa:

```text
data/ejemplo_ctg_procesado.csv
```

## Probar datos maternos

Usa los valores de ejemplo de:

```text
data/ejemplo_materno.csv
```
