import sys
from pathlib import Path

# Permite importar AG_procesamiento desde las pruebas y desde AppTest
# (streamlit run agrega la carpeta del script al path, AppTest no).
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
