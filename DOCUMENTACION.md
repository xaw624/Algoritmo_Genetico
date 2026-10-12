# Documentación técnica — Simulador de Algoritmo Genético 1D

Aplicación web en [Streamlit](https://streamlit.io/) que minimiza una función real de una variable `f(x)` en un intervalo `[a, b]` con un **algoritmo genético (AG)** de codificación real, y anima la evolución de la población generación por generación.

---

## 1. Estructura del proyecto

| Archivo | Rol |
|---|---|
| `app.py` | Interfaz Streamlit: parámetros, ejemplos, animación y resultados. |
| `AG_procesamiento.py` | Motor del AG (`GeneticAlgorithm1D`), validación de expresiones (`validar_expresion`) y parser (`build_function`). No depende de Streamlit. |
| `tests/` | Pruebas `pytest`: motor (`test_ag_procesamiento.py`) y app de punta a punta con `AppTest` (`test_app.py`). |
| `requirements.txt` | Dependencias directas de la app, fijadas a versiones probadas (UTF-8). |
| `requirements-dev.txt` | `requirements.txt` + `pytest`. |
| `.gitattributes` | `* text=auto`: normaliza finales de línea a LF en el repo. |
| `deploy/` | Instalación en un servidor dedicado: script `instalar.sh`, plantilla systemd, lanzador, variables y límites (ver §9). |
| `.streamlit/config.toml` | Tema y ping del websocket. Se aplica al correr Streamlit desde la carpeta del repo. |
| `Caminata aleatoria1v.py` | Versión antigua de escritorio en Tkinter (caminata aleatoria). Está en `.gitignore` y no la usa la app web. |
| `.python-version` | Lo genera `pyenv local`. Está en `.gitignore` porque el nombre del entorno depende de cada máquina. |

---

## 2. Entorno de desarrollo (pyenv)

Entorno creado con **Python 3.12.14** usando `pyenv-virtualenv`. Se requiere Python ≥ 3.11, porque `numpy 2.3` no soporta versiones anteriores.

```bash
pyenv virtualenv 3.12.14 algoritmo-genetico-env
pyenv local algoritmo-genetico-env
pip install -r requirements-dev.txt     # o requirements.txt si no vas a correr pruebas
streamlit run app.py                    # http://localhost:8501
pytest                                  # 60 pruebas, ~20 s
```

**`requirements.txt`**: contiene solo las 4 dependencias directas (`streamlit 1.63.0`, `numpy 2.3.0`, `matplotlib 3.10.8`, `sympy 1.14.0`). Streamlit instala por su cuenta pandas, pyarrow, altair, etc. Al no fijar las dependencias indirectas, una instalación nueva puede traer versiones más recientes. Si se necesita reproducibilidad exacta, conviene generar un lock (`pip freeze > requirements-lock.txt`, o usar `pip-tools` / `uv`).

---

## 3. Flujo de ejecución

Streamlit **vuelve a ejecutar `app.py` de arriba a abajo** con cada interacción. Para que los resultados sobrevivan a esos reruns, se guardan en `st.session_state.resultado`.

```mermaid
flowchart TD
    A[Usuario interactúa] -->|rerun| B[app.py se ejecuta completo]
    B --> C{¿Se pulsó ▶️ Ejecutar?}
    C -- Sí --> E[Validar dominio]
    E --> F[build_function: validar AST → SymPy evaluate=False → lambdify]
    F --> G[Muestrear 600 puntos de f para la curva]
    G --> H[GeneticAlgorithm1D.run]
    H --> I[Guardar resultado en session_state]
    I --> J[mostrar_resultados con animar=True]
    C -- No --> K{¿Hay resultado guardado?}
    K -- Sí --> L[mostrar_resultados con animar=False: generación del slider]
    K -- No --> M[Solo UI, ayuda y ejemplos]
    E -- error --> X[Borrar resultado anterior y mostrar error]
    F -- error --> X
```

---

## 4. Motor: `AG_procesamiento.py`

### 4.1 Validación y parseo de expresiones

`sympy.sympify` usa `eval` internamente, así que la expresión del usuario **nunca** llega a SymPy sin validarse antes.

**`validar_expresion(expr_str)`** ([AG_procesamiento.py:93](AG_procesamiento.py:93)):

1. Rechaza expresiones de más de `MAX_LONGITUD_EXPRESION = 200` caracteres.
2. Analiza el texto con `ast.parse(..., mode="eval")`. Un error de sintaxis se convierte en `ValueError("Sintaxis inválida: ...")`.
3. Recorre el árbol y acepta solo una lista blanca:
   - números `int`/`float` (no `bool`, `complex` ni cadenas);
   - los nombres `x`, `pi`, `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `Abs`;
   - los operadores `+ - * / **`, `^` (SymPy lo convierte en potencia) y el signo unario;
   - llamadas solo a las funciones de la lista y sin argumentos con nombre.

   Cualquier otra cosa se rechaza: atributos (`().__class__`), `lambda`, listas, `__import__`, otras variables, etc.
4. **`_valor_constante`** evalúa con `float` cada subárbol que no depende de `x`. Si alguno desborda (`9**9**9`, `(9**100)**100`, `exp(1000)`), lanza `ValueError("...constante demasiado grande")`. Sin esto, SymPy intentaba calcular enteros de cientos de millones de dígitos y bloqueaba el servidor.

**`build_function(expr_str)`** ([AG_procesamiento.py:377](AG_procesamiento.py:377)):

1. Llama a `validar_expresion`.
2. `sympy.sympify(..., locals=permitidos, evaluate=False)`. Con `evaluate=False`, SymPy **no simplifica**. Sin esa opción, `9**(x - x + 9**9)` se reducía a `9**387420489` y volvía a bloquear el proceso.
3. Rechaza símbolos libres distintos de `x`.
4. `sympy.lambdify(x, expr, "numpy")` y un envoltorio con `np.errstate(all="ignore")`.

### 4.2 `GeneticAlgorithm1D`

**Representación:** cada individuo es un real `x ∈ [a, b]`. El problema es de **minimización**; internamente el fitness es `-f(x)`.

| Parámetro | Por defecto (clase / app) | En la UI | Significado |
|---|---|---|---|
| `f` | — | Función f(x) | Función objetivo |
| `bounds` | — / `(-10, 10)` | Límites del dominio | Dominio `[a, b]` |
| `pop_size` | 30 / 50 | Tamaño de población | Individuos por generación |
| `pc` | 0.8 | pc | Probabilidad de crossover |
| `pm` | 0.1 | pm | Probabilidad de mutación |
| `sigma` | 0.1 | Sigma relativo | `σ₀ = sigma·(b−a)` |
| `generations` | 200 / 80 | Cantidad de generaciones | Iteraciones evolutivas |
| `tournament_k` | 3 | Avanzados → k | Tamaño del torneo |
| `sigma_decay` | 0.99 | Avanzados → γ | `σ_t = σ₀·γᵗ` |
| `random_state` | `None` / 42 | Fijar semilla + Semilla | Semilla de `np.random.default_rng`; `None` = aleatoria |

**Ciclo de `run()`** ([AG_procesamiento.py:331](AG_procesamiento.py:331)):

1. **Reinicio** del estado, para poder reutilizar el objeto.
2. **Inicialización:** `pop_size` valores uniformes en `[a, b]`.
3. **Evaluación** (`evaluate`): `f` en cada individuo. Los errores, `nan` e `inf` se convierten en `+inf` (penalización). Actualiza el mejor histórico y guarda `history`, `values_history` y `best_history`.
4. Si **ningún** individuo de la generación 0 tiene `f(x)` finito, lanza un `ValueError` claro (“La función no tiene valores finitos en ningún individuo de la población inicial…”).
5. Repite `generations` veces:
   - **Elitismo:** el mejor histórico pasa directo.
   - **Torneo:** `k` individuos distintos al azar; gana el de mayor fitness.
   - **Crossover aritmético:** con prob. `pc`, `hijo = α·p1 + (1−α)·p2`, `α ~ U(0,1)`; si no, `hijo = p1`.
   - **Mutación gaussiana:** con prob. `pm`, `x += N(0, σ_t)`; después siempre se aplica `np.clip` a `[a, b]`.
   - `σ ← σ·γ`, luego se vuelve a evaluar.
6. Devuelve `(best_x, best_f, history, best_history)`. Además, `ag.values_history` contiene los `f(x)` de cada población, ya calculados, para que la app no tenga que volver a evaluarlos.

**Costo:** `pop_size × (generations + 1)` evaluaciones de `f` (≈ 4 050 por defecto, ~0.35 s).

---

## 5. Interfaz: `app.py`

### 5.1 Componentes

| Zona | Contenido |
|---|---|
| Barra lateral | Función, dominio, población (4–500), generaciones (1–500), `pc`, `pm`, `sigma`; expander **🔧 Parámetros avanzados** (`k`, `γ`); velocidad de animación; casilla **Fijar semilla aleatoria** y semilla (deshabilitada si la casilla está desmarcada); botón ▶️. Cada control tiene `key` para poder probarlo. |
| Expanders | “Instrucciones de uso” (sintaxis, límite de 200 caracteres) y “Fundamento teórico” (LaTeX: torneo con `k`, `σ_t = σ₀γᵗ`, parámetros avanzados, semilla opcional). |
| Ejemplos rápidos | Lista `EJEMPLOS` con 12 pares (etiqueta, expresión), dibujados en filas de 4. Cada botón usa `on_click=set_example`. |
| Resultados | Leyenda con la función, el dominio y la semilla usados; 3 métricas; animación de la población; slider **Explorar generación**; gráfica de convergencia con una línea punteada en la generación seleccionada; botón **🗑️ Limpiar resultados**. |

### 5.2 Funciones auxiliares

| Función | Rol |
|---|---|
| `evaluar_vector_seguro(f, xs)` | Evalúa la curva punto a punto; los errores se convierten en `NaN`. |
| `format_value(v)` | 6 decimales o `"N/A"`. |
| `limpiar_resultados()` | Borra `st.session_state.resultado`. |
| `crear_figura(resultado)` | Crea la figura, calcula los márgenes una vez y guarda el **fondo** (curva, ejes, rejilla, leyenda) para hacer *blitting*. |
| `renderizar_generacion(figura, resultado, gen)` | Restaura el fondo, repinta solo los puntos, la ★, la leyenda y el título, y devuelve un arreglo RGB para `st.image`. |
| `texto_estado(resultado, gen)` | “Generación g/G \| Mejor histórico: …”. |
| `mostrar_resultados(resultado, animar, frame_delay)` | Dibuja la sección de resultados. Con `animar=True` reproduce como máximo `MAX_FRAMES_ANIMACION = 100` frames espaciados (siempre la generación 0 y la última). Si no, muestra la generación del slider. |

El slider usa la key `gen_<id de ejecución>`, de modo que cada simulación nueva empieza en la última generación.

---

## 6. Verificación

**Pruebas automatizadas** (`pytest`, 60 pruebas, todas pasan):

- `test_ag_procesamiento.py`:
  - Los 12 ejemplos y variantes (`x^2`, espacios iniciales, `Abs`) se aceptan.
  - 13 entradas maliciosas o inválidas se rechazan.
  - Las constantes enormes se rechazan y `9**(x - x + 9**9)` ya no bloquea el proceso.
  - El AG encuentra el mínimo real (malla de 200 001 puntos, tolerancia 10⁻²) en los 12 ejemplos.
  - Los historiales quedan alineados y dentro del dominio.
  - El elitismo garantiza que el mejor histórico nunca empeora, y la semilla hace los resultados reproducibles.
  - `sigma_decay` y `tournament_k` se aplican, y los casos de error dan mensajes claros.
- `test_app.py` (con `AppTest`):
  - La ejecución por defecto funciona.
  - Los resultados persisten al mover un control.
  - El slider de generación y el botón de limpiar funcionan.
  - Los 12 botones de ejemplo producen expresiones válidas.
  - La ejecución sin semilla funciona.
  - Un dominio inválido borra los resultados.
  - Una expresión maliciosa **no** llega a ejecutarse: se comprueba que no crea un archivo.

**Resultados del AG** (dominio `[-10, 10]`, 50 individuos, 80 generaciones, semilla 42):

| Ejemplo | best_x | best_f | mínimo real |
|---|---:|---:|---:|
| Parábola | 0.0000 | 0.0000 | 0.0000 |
| Seno | −3.7499 | −1.0000 | −1.0000 |
| Gaussiana | 10.0000 | 0.0000 | 0.0000 |
| Valor absoluto | 0.0000 | 0.0000 | 0.0000 |
| Doble pozo | 1.0000 | 0.0000 | 0.0000 |
| Múltiples mínimos | 9.8970 | −9.8965 | −9.8965 |
| Logarítmica | 0.0000 | 0.0000 | 0.0000 |
| Sinusoidal | 4.7124 | −1.0000 | −1.0000 |
| Pendiente con oscilación | −8.9129 | −1.8907 | −1.8907 |
| Valles múltiples | −7.9981 | −1.9887 | −1.9887 |
| Terreno irregular | 6.0752 | −0.9985 | −0.9985 |
| Escalón sigmoide | −10.0000 | 0.0000 | 0.0000 |

**Rendimiento** (corrida por defecto, 80 generaciones, animación a 10 ms, máquina de 2 núcleos):

| | Antes | Ahora |
|---|---:|---:|
| Dibujo por frame | ~215 ms (`savefig`, dpi 120, `bbox_inches="tight"`) | ~65 ms (blitting) |
| Corrida completa con animación | 21.4 s | 8.3 s (7.7 s con Streamlit 1.63) |
| Cambiar de generación con el slider | — (no existía) | ~0.6 s |

---

## 7. Cambios aplicados

| Problema detectado | Corrección |
|---|---|
| 🔴 **Ejecución de código arbitrario** desde el campo f(x): `sympify` usa `eval` | Lista blanca sobre el AST (`validar_expresion`) antes de llamar a SymPy. |
| 🔴 **Bloqueo del servidor** con `9**9**9` o `9**(x - x + 9**9)` | `_valor_constante` acota las constantes y se usa `sympify(..., evaluate=False)`. |
| 🔴 Población y generaciones sin máximo: cualquiera podía lanzar corridas de horas | Límite de 500 en cada una (peor caso 500 × 500 ≈ 26 s). |
| 🔴 `sympy` faltaba en `requirements.txt`, que además estaba en UTF-16 con 44 paquetes fijados | Regenerado en UTF-8 con las 4 dependencias directas; nuevo `requirements-dev.txt`. |
| 🟠 Ejemplo “Doble pozo” con un paréntesis de más | `"(x**2 - 1)**2"`. Los ejemplos ahora son una lista `EJEMPLOS` (100 líneas repetidas → un bucle). |
| 🟠 “🔄 Ejecutar otra simulación” en realidad borraba los resultados | Se reemplazó por **🗑️ Limpiar resultados**, que hace lo que dice. |
| 🟠 Los resultados desaparecían al mover cualquier control | Se guardan en `st.session_state`; los reruns los muestran sin repetir la animación. |
| 🟠 `RuntimeError` engañoso si ningún individuo inicial era válido | `ValueError` con un mensaje claro en `run()`. |
| 🟠 La semilla decía “(opcional)” pero siempre se usaba | Casilla **Fijar semilla aleatoria**; desmarcada pasa `None`. |
| 🟡 El fundamento teórico describía `σ` constante y no mencionaba `k` | Texto actualizado (`σ_t = σ₀γᵗ`, torneo con `k`); `γ` y `k` expuestos en “Parámetros avanzados”. |
| 🟡 Animación lenta | Blitting, como máximo 100 frames, métricas antes de la animación; se dejó de evaluar `f` dos veces (`values_history`). |
| 🟡 Tras un error, seguían visibles los resultados anteriores | Un error borra el resultado guardado. |
| 🟡 README desactualizado, cortado y con enlace a otro repo | Reescrito para el AG, con instalación, uso y pruebas. |
| 🟢 `__pycache__/` versionado y ruido CRLF/LF en `git status` | `.gitignore` actualizado y `.gitattributes` con `* text=auto`. Los archivos editados quedaron en LF, como en el repo. |
| 🟢 Sin pruebas | 60 pruebas `pytest`. |
| 🟢 Streamlit 1.45.1, con `use_container_width` en vías de desaparecer | Actualizado a **Streamlit 1.63.0** (y matplotlib 3.10.8): `use_container_width=True` → `width="stretch"`. El mensaje de éxito usa `icon="✅"`, porque 1.63 separa el emoji inicial del texto. Las pruebas corren sin avisos de deprecación. |

---

## 8. Recomendaciones pendientes

Ninguna es un error; son mejoras opcionales:

1. **Animación en el navegador con Plotly.** Una figura con `frames` se reproduciría en el cliente, con play/pausa, sin reruns ni imágenes enviadas desde el servidor. Requiere agregar `plotly`.
2. **Operadores del AG:**
   - **BLX-α** (`α ∈ [−0.5, 1.5]`) en vez del crossover aritmético, que solo genera hijos *entre* los padres y contrae la población.
   - **Reflexión** en los bordes en vez de `np.clip`, que acumula individuos exactamente en los límites.
   - **Parada temprana** si el mejor valor no mejora en `n` generaciones.
3. **Lock de dependencias** si se necesita reproducibilidad exacta del entorno (ver §2).
4. **Vectorizar la evaluación.** `evaluate_individual` llama a `f` punto a punto (~85 µs por individuo). Evaluar la población completa como un arreglo de NumPy (con respaldo punto a punto si falla) acelera el motor varias veces y permitiría subir los límites de 500.

---

## 9. Despliegue en producción

La app corre como servicio systemd `streamlit@genetico`, que escucha solo en `127.0.0.1:8516`. Un **Cloudflare Tunnel** la publica con HTTPS en **https://genetico.streamlytics.stream** sin abrir puertos de entrada. El destino es un servidor dedicado: una instancia **AWS EC2** con Ubuntu Server 24.04 LTS.

```mermaid
flowchart LR
    U[Navegador] -->|HTTPS| CF[Cloudflare<br/>DNS genetico → túnel, con proxy]
    CF -->|túnel saliente| CD[cloudflared<br/>en la instancia]
    CD -->|http://127.0.0.1:8516| S[streamlit@genetico]
    S --> A[app.py + .venv]
```

### 9.1 Archivos

| Archivo | Destino en el servidor | Función |
|---|---|---|
| `deploy/instalar.sh` | — | Instala todo: paquetes, swap, `.venv`, servicio y, opcionalmente, `cloudflared`. Se puede volver a ejecutar para actualizar. |
| `deploy/streamlit@.service` | `/etc/systemd/system/` | Plantilla systemd. Su `WorkingDirectory` es el repo, para que se lea `.streamlit/config.toml`. |
| `deploy/run_streamlit.sh` | `/usr/local/bin/` | Lanzador. Fija `--server.address=127.0.0.1`. |
| `deploy/genetico.env` | `/etc/streamlit/genetico.env` (`root:ubuntu`, `640`) | Python de `.venv`, `app.py`, puerto 8516 y `--client.showErrorDetails=none` (no muestra tracebacks a los visitantes). |
| `deploy/limites.conf` | `/etc/systemd/system/streamlit@genetico.service.d/` | Como máximo 1 núcleo, `MemoryHigh=400M`, `MemoryMax=512M`. |
| `.streamlit/config.toml` | (en el repo) | Tema y `websocketPingInterval = 30`, para que Cloudflare no cierre el websocket. |

> `instalar.sh` se niega a correr si ya existe una plantilla `streamlit@.service` distinta (un servidor compartido con otras apps), para no afectarlas.

### 9.2 Crear la instancia EC2 (consola de AWS)

Región **us-east-2 (Ohio)**. *EC2 → Lanzar instancia*:

| Campo | Valor |
|---|---|
| Nombre | `genetico` |
| Imagen (AMI) | **Ubuntu Server 24.04 LTS**, arquitectura **64 bits (Arm)**. El script requiere Python 3.11–3.13; versiones más nuevas de Ubuntu traen un Python para el que no hay builds de las dependencias fijadas. |
| Tipo de instancia | `t4g.micro` (2 vCPU, 1 GB) |
| Par de claves | No hace falta si te conectas con EC2 Instance Connect. |
| Red | VPC predeterminada, con *Asignar IP pública automáticamente* **habilitado**: la instancia necesita salir a Internet. |
| Grupo de seguridad | Uno nuevo, con **una sola regla de entrada**: SSH (22) con origen la lista de prefijos `com.amazonaws.us-east-2.ec2-instance-connect`, para conectarte desde la consola. El tráfico web entra por el túnel, así que no hace falta abrir nada más. |
| Almacenamiento | 8–20 GB gp3 |
| Detalles avanzados → *Especificación de crédito* | **Estándar**: si se agotan los créditos de CPU, la instancia se ralentiza en lugar de cobrar extra. |

### 9.3 Instalar la app

*Instancia → Conectar → EC2 Instance Connect → Conectar* abre una terminal en el navegador como el usuario `ubuntu`:

```bash
git clone https://github.com/xaw624/Algoritmo_Genetico.git ~/Algoritmo_Genetico
bash ~/Algoritmo_Genetico/deploy/instalar.sh
```

Tarda unos minutos, sobre todo por `pip`. Debe terminar con `La app responde en http://127.0.0.1:8516`.

### 9.4 Conectar el Cloudflare Tunnel

El servidor usa su propio túnel:

1. En Cloudflare: *Zero Trust → Networks → Tunnels → Create a tunnel* → tipo **Cloudflared** → nombre `genetico-aws`.
2. En *Install and run a connector*, elige Debian / arm64 y copia el **token**: la cadena larga que aparece después de `service install`. Es una credencial: no la compartas ni la guardes en el repo.
3. En la instancia, ejecuta el comando **empezando con un espacio**; así no queda en el historial de bash:

   ```bash
    CLOUDFLARE_TUNNEL_TOKEN='<token>' bash ~/Algoritmo_Genetico/deploy/instalar.sh
   ```

   El túnel debe aparecer como *Healthy* en Cloudflare.
4. **Probar con un hostname temporal.** En el túnel, *Public hostname* (o *Published application routes*): `genetico-aws.streamlytics.stream` → tipo `HTTP` → `127.0.0.1:8516`. Ábrelo en el navegador y ejecuta una simulación.
5. **Pasar el dominio definitivo:**
   - En DNS, borra el registro `genetico` que apunta al túnel anterior.
   - En el túnel nuevo, agrega `genetico.streamlytics.stream` → `http://127.0.0.1:8516`. Cloudflare crea el registro DNS.
   - Entre esos dos pasos el sitio queda fuera unos segundos.
   - Al final, borra el hostname temporal.

### 9.5 Retirar la app del servidor anterior

Cuando el dominio ya responda desde AWS, ejecuta en el servidor anterior:

```bash
sudo systemctl disable --now streamlit@genetico
sudo rm -r /etc/streamlit/genetico.env /etc/systemd/system/streamlit@genetico.service.d
sudo systemctl daemon-reload
```

La regla de `genetico` del túnel anterior deja de usarse en cuanto el DNS apunta al túnel nuevo. Se puede quitar de su `config.yml` en el siguiente mantenimiento; reiniciar `cloudflared` corta un momento las demás webs de ese túnel.

### 9.6 Operación

```bash
systemctl status streamlit@genetico --no-pager -l                 # estado, memoria y límites
journalctl -u streamlit@genetico -f                                # logs en vivo
curl http://127.0.0.1:8516/_stcore/health                          # salud local → "ok"
git -C ~/Algoritmo_Genetico pull && bash ~/Algoritmo_Genetico/deploy/instalar.sh   # actualizar
```

Para cambiar los límites o las variables, edita `deploy/limites.conf` o `deploy/genetico.env` y vuelve a ejecutar `instalar.sh`.

**Desinstalar:**

```bash
sudo systemctl disable --now streamlit@genetico cloudflared
sudo cloudflared service uninstall
sudo rm -r /etc/streamlit /etc/systemd/system/streamlit@genetico.service.d \
           /etc/systemd/system/streamlit@.service /usr/local/bin/run_streamlit.sh
sudo systemctl daemon-reload
```

### 9.7 Costos y notas

- **Costo aproximado:** ~US$10 al mes = `t4g.micro` (~US$6) + IP pública (~US$3.6) + disco (~US$0.6–1.6). Son precios orientativos de us-east-2; confírmalos en la calculadora de AWS. Conviene crear un presupuesto en AWS Budgets con alerta.
- **Consumo medido** (Streamlit 1.63): ~40 MB en reposo y ~180 MB de pico en una corrida de 500 × 500. El script agrega 1 GB de swap en instancias con menos de 2 GB de RAM.
- **Verificación pública:** Cloudflare puede bloquear con `403` las peticiones que salen del propio servidor (WAF). La URL pública se comprueba desde un navegador; desde el servidor bastan el `curl` local y el estado del túnel en Cloudflare.
- **Errores:** en producción se ocultan los tracebacks (`genetico.env`). Con `streamlit run app.py` en local se muestran completos.
- **Sin login:** es un simulador educativo público y no maneja datos.
