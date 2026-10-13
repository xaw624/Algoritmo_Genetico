# Simulador de Algoritmo Genético 1D

Aplicación web interactiva construida con [Streamlit](https://streamlit.io/) que minimiza una función matemática `f(x)` en un intervalo `[a, b]` mediante un **algoritmo genético** y anima la evolución de la población generación por generación.

## 🚀 Características

- Ingreso de funciones personalizadas `f(x)` (`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pi`), validadas de forma segura y en vivo: la fórmula interpretada se muestra en LaTeX y los errores aparecen junto al campo.
- 13 ejemplos que cargan la función con su dominio sugerido y una descripción: funciones multimodales, con mínimos en el borde, sin derivada en el mínimo, etc.
- Vista previa de f(x) antes de ejecutar.
- Control de todos los parámetros del AG: población, generaciones, `pc`, `pm`, `σ` relativo, enfriamiento de `σ`, tamaño del torneo y semilla.
- Animación de la población y del mejor individuo histórico (lenta, normal, rápida u omitida), con un control para revisar cualquier generación.
- Aviso cuando los resultados mostrados ya no corresponden a los parámetros actuales.
- Gráfica de convergencia del mejor valor encontrado.
- Explicación teórica del método (en LaTeX) dentro de la app.

## ⚙️ Instalación

Requiere **Python 3.11 o superior**.

1. Clona este repositorio:

   ```bash
   git clone https://github.com/xaw624/Algoritmo_Genetico.git
   cd Algoritmo_Genetico
   ```

2. Crea un entorno virtual. Con [pyenv](https://github.com/pyenv/pyenv) y `pyenv-virtualenv`:

   ```bash
   pyenv virtualenv 3.12.14 algoritmo-genetico-env
   pyenv local algoritmo-genetico-env
   ```

   O con `venv`:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # En Windows: .venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Uso

```bash
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

En producción está publicada en **https://genetico.streamlytics.stream** como servicio `streamlit@genetico` (ver [DOCUMENTACION.md §9](DOCUMENTACION.md)).

## 🧪 Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

## 📁 Estructura

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz de Streamlit |
| `AG_procesamiento.py` | Algoritmo genético y validación/parseo de expresiones |
| `tests/` | Pruebas del algoritmo y de la app (`pytest`) |
| `deploy/` | Instalación en un servidor dedicado (AWS EC2): script, servicio systemd y límites |
| `DOCUMENTACION.md` | Documentación técnica detallada (incluye el despliegue en producción) |

## 📄 Licencia

[MIT](LICENSE.txt) © 2025 Jesús Uriel Moguel Bautista
