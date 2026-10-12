import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from AG_procesamiento import MAX_LONGITUD_EXPRESION, GeneticAlgorithm1D, build_function


# Máximo de imágenes en la animación. Con más generaciones se muestran
# generaciones espaciadas uniformemente (siempre la 0 y la última).
MAX_FRAMES_ANIMACION = 100

# Ejemplos rápidos: (texto del botón, expresión)
EJEMPLOS = [
    ("Parábola", "x**2"),
    ("Seno", "sin(2*pi*x/3)"),
    ("Gaussiana", "exp(-x**2)"),
    ("Valor absoluto", "abs(x)"),
    ("Doble pozo", "(x**2 - 1)**2"),
    ("Múltiples mínimos", "x*sin(10*x)"),
    ("Logarítmica", "log(abs(x) + 1)"),
    ("Sinusoidal", "sin(5*x)"),
    ("Pendiente con oscilación", "0.1*x + sin(3*x)"),
    ("Valles múltiples", "sin(x) + sin(10*x/3)"),
    ("Terreno irregular", "0.3*sin(8*x) + 0.7*cos(15*x)"),
    ("Escalón sigmoide", "1/(1 + exp(-10*x))"),
]


# -----------------------------------------------------------
# Configuración general de la página
# -----------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Algoritmo Genético 1D",
    layout="wide",
)

st.title("Simulador de Algoritmo Genético 1D")


# -----------------------------------------------------------
# Estado de sesión para la función actual
# Esto permite que los ejemplos rápidos sí actualicen el input.
# -----------------------------------------------------------
if "func" not in st.session_state:
    st.session_state.func = "x**2 + 10*cos(2*x) + x*sin(5*x)"


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def evaluar_vector_seguro(f, x_values):
    """
    Evalúa un arreglo de x de forma segura.
    Si en algún punto la función falla o devuelve algo no finito,
    ese valor se marca como NaN para no romper la gráfica.
    """
    resultados = []

    for x in np.asarray(x_values, dtype=float):
        try:
            with np.errstate(all="ignore"):
                y = f(float(x))

            arr = np.asarray(y, dtype=float)

            # Esperamos un escalar por cada x
            if arr.size != 1:
                resultados.append(np.nan)
                continue

            valor = float(arr.reshape(-1)[0])

            if np.isfinite(valor):
                resultados.append(valor)
            else:
                resultados.append(np.nan)

        except Exception:
            resultados.append(np.nan)

    return np.array(resultados, dtype=float)


def set_example(expr):
    """
    Cambia la función actual desde los botones de ejemplo.
    Al usar on_click, Streamlit vuelve a ejecutar la app automáticamente.
    """
    st.session_state.func = expr

def format_value(value, decimals=6):
    """
    Formatea un número de forma segura para mostrarlo en métricas.
    """
    try:
        if value is None:
            return "N/A"

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{decimals}f}"
    except Exception:
        return "N/A"


def limpiar_resultados():
    """
    Borra los resultados guardados de la última simulación.
    """
    st.session_state.pop("resultado", None)


def crear_figura(resultado):
    """
    Crea la figura con la curva de f(x) y devuelve los elementos
    que cambian entre generaciones.
    """
    x_plot = resultado["x_plot"]
    y_plot = resultado["y_plot"]
    valid_curve = np.isfinite(y_plot)

    # Rango del eje Y: curva + todos los valores válidos de la población
    all_y_values = [y_plot[valid_curve]]
    for values in resultado["values_history"]:
        all_y_values.append(values[np.isfinite(values)])
    all_y_values = np.concatenate(all_y_values)

    y_min = float(all_y_values.min())
    y_max = float(all_y_values.max())
    y_range = max(y_max - y_min, 1e-6)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Curva de la función
    ax.plot(
        x_plot[valid_curve],
        y_plot[valid_curve],
        linewidth=2,
        alpha=0.75,
        label=f"f(x) = {resultado['expr']}",
    )

    # Scatter de la población actual
    population_scatter = ax.scatter([], [], s=55, zorder=4, label="Población")

    # Scatter del mejor histórico
    best_scatter = ax.scatter([], [], s=160, marker="*", zorder=5, label="Mejor histórico")

    ax.set_xlim(*resultado["bounds"])
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Algoritmo Genético 1D - Generación 0")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # Márgenes calculados una sola vez (con título, para reservarle espacio)
    fig.tight_layout()

    # Blitting: el fondo (curva, ejes, rejilla, leyenda) se dibuja una sola vez
    # y en cada frame solo se repintan los puntos y el título.
    # Redibujar la figura completa por frame era lo más lento de la animación.
    ax.set_title("")
    fig.canvas.draw()
    fondo = fig.canvas.copy_from_bbox(fig.bbox)

    return fig, ax, population_scatter, best_scatter, fondo


def renderizar_generacion(figura, resultado, gen):
    """
    Actualiza la figura con la población de la generación `gen`
    y la devuelve como arreglo RGB para st.image.
    """
    fig, ax, population_scatter, best_scatter, fondo = figura
    fig.canvas.restore_region(fondo)

    # Población visible (solo los individuos con f(x) finito)
    population = resultado["history"][gen]
    pop_y = resultado["values_history"][gen]
    valid_pop = np.isfinite(pop_y)
    population_scatter.set_offsets(np.column_stack((population[valid_pop], pop_y[valid_pop])))

    # Mejor histórico
    best_x_hist, best_f_hist = resultado["best_history"][gen]
    best_scatter.set_offsets([[best_x_hist, best_f_hist]])

    ax.set_title(f"Algoritmo Genético 1D - Generación {gen}")

    # La leyenda se repinta al final para que quede encima de los puntos
    for artista in (population_scatter, best_scatter, ax.get_legend(), ax.title):
        fig.draw_artist(artista)

    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def texto_estado(resultado, gen):
    """
    Texto con la generación mostrada y el mejor histórico hasta ella.
    """
    total = len(resultado["history"]) - 1
    best_x_hist, best_f_hist = resultado["best_history"][gen]
    return (
        f"Generación {gen}/{total}  |  "
        f"Mejor histórico: x = {best_x_hist:.6f}, f(x) = {best_f_hist:.6f}"
    )


def mostrar_resultados(resultado, animar, frame_delay):
    """
    Dibuja los resultados guardados en session_state.
    Si animar=True (la simulación se acaba de ejecutar) reproduce la animación;
    si no, muestra la generación elegida en el slider sin repetirla.
    """
    total_gens = len(resultado["history"]) - 1
    a, b = resultado["bounds"]

    # -------------------------------------------------------
    # Resultados finales
    # -------------------------------------------------------
    st.subheader("📊 Resultados")
    st.caption(
        f"f(x) = `{resultado['expr']}` en [{a}, {b}]  ·  semilla: {resultado['semilla']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mejor x encontrado", format_value(resultado["best_x"]))

    with col2:
        st.metric("Mejor f(x)", format_value(resultado["best_f"]))

    with col3:
        st.metric("Generaciones evaluadas", str(total_gens))

    # -------------------------------------------------------
    # Población generación por generación
    # -------------------------------------------------------
    st.subheader("🧬 Evolución de la población")

    figura = crear_figura(resultado)
    imagen = st.empty()
    estado = st.empty()

    if animar:
        progress_bar = st.progress(0)
        frames = np.unique(
            np.linspace(0, total_gens, min(total_gens + 1, MAX_FRAMES_ANIMACION)).round().astype(int)
        )

        for i, gen in enumerate(frames):
            imagen.image(
                renderizar_generacion(figura, resultado, gen),
                output_format="PNG",
                width="stretch",
            )
            estado.text(texto_estado(resultado, gen))
            progress_bar.progress((i + 1) / len(frames))

            # Pausa para controlar velocidad de animación
            time.sleep(frame_delay / 1000)

        progress_bar.empty()
        st.success("Simulación completada.", icon="✅")

    # La key cambia en cada ejecución para que el slider arranque en la última generación
    gen = st.slider(
        "Explorar generación",
        min_value=0,
        max_value=total_gens,
        value=total_gens,
        key=f"gen_{resultado['id']}",
        help="Muestra la población de cualquier generación sin repetir la animación.",
    )

    if not animar:
        imagen.image(
            renderizar_generacion(figura, resultado, gen),
            output_format="PNG",
            width="stretch",
        )
        estado.text(texto_estado(resultado, gen))

    plt.close(figura[0])
    st.caption("La generación 0 corresponde a la población inicial.")

    # -------------------------------------------------------
    # Gráfica extra: evolución del mejor histórico
    # -------------------------------------------------------
    st.subheader("📉 Evolución del mejor valor encontrado")

    best_values = [best_f_hist for _, best_f_hist in resultado["best_history"]]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(best_values, marker="o", linewidth=1.5)
    ax2.axvline(gen, color="gray", linestyle=":", alpha=0.8)
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Mejor f(x) histórico")
    ax2.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig2)
    plt.close(fig2)

    st.button("🗑️ Limpiar resultados", width="stretch", on_click=limpiar_resultados)


# -----------------------------------------------------------
# Panel de ayuda
# -----------------------------------------------------------
with st.expander("ℹ️ Instrucciones de uso", expanded=False):
    st.markdown(
        f"""
        **Sintaxis para ecuaciones:**
        - Variable: `x`
        - Operadores: `+`, `-`, `*`, `/`, `**` (también `^` para potencias)
        - Funciones:
            - `sin(x)`, `cos(x)`, `tan(x)`
            - `exp(x)`, `log(x)`
            - `sqrt(x)`, `abs(x)`
        - Constante:
            - `pi`
        - Longitud máxima: {MAX_LONGITUD_EXPRESION} caracteres.

        **Qué hace esta app:**
        - Genera una población inicial aleatoria dentro del dominio.
        - Evalúa qué tan buena es cada solución.
        - Aplica selección, crossover, mutación y elitismo.
        - Muestra la evolución generación por generación.

        **Interpretación visual:**
        - Los puntos representan la población.
        - La estrella representa el mejor valor histórico encontrado.
        - La generación 0 es la población inicial.
        - Al terminar la animación, el control *Explorar generación* permite revisar cualquier generación.
        - Los resultados se conservan hasta la siguiente simulación o hasta pulsar *Limpiar resultados*.
        """
    )


# -----------------------------------------------------------
# Sidebar: parámetros del AG
# -----------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del algoritmo")

eq_str = st.sidebar.text_input(
    "Función f(x):",
    key="func",
    help="Escribe la función a minimizar usando la variable x.",
)

x_min = st.sidebar.number_input("Límite inferior del dominio", value=-10.0, key="x_min")
x_max = st.sidebar.number_input("Límite superior del dominio", value=10.0, key="x_max")

pop_size = st.sidebar.number_input(
    "Tamaño de población",
    min_value=4,
    max_value=500,
    value=50,
    step=1,
    key="pop_size",
    help="Cantidad de individuos por generación.",
)

generations = st.sidebar.number_input(
    "Cantidad de generaciones",
    min_value=1,
    max_value=500,
    value=80,
    step=1,
    key="generations",
)

pc = st.sidebar.slider(
    "Probabilidad de crossover (pc)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
    key="pc",
)

pm = st.sidebar.slider(
    "Probabilidad de mutación (pm)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    key="pm",
)

sigma = st.sidebar.number_input(
    "Sigma relativo de mutación",
    min_value=0.001,
    max_value=1.0,
    value=0.10,
    step=0.01,
    key="sigma",
    help="Se multiplica por el ancho del dominio. Ejemplo: 0.10 equivale al 10% del rango.",
)

with st.sidebar.expander("🔧 Parámetros avanzados"):
    tournament_k = st.number_input(
        "Tamaño del torneo (k)",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        key="tournament_k",
        help="Individuos que compiten en cada selección. Un k mayor aumenta la presión selectiva.",
    )

    sigma_decay = st.slider(
        "Enfriamiento de sigma (γ)",
        min_value=0.90,
        max_value=1.00,
        value=0.99,
        step=0.005,
        format="%.3f",
        key="sigma_decay",
        help="En cada generación sigma se multiplica por este factor. 1.000 mantiene sigma constante.",
    )

frame_delay = st.sidebar.slider(
    "Velocidad de animación (ms)",
    min_value=10,
    max_value=500,
    value=80,
    key="frame_delay",
    help=(
        "Tiempo entre generaciones en la animación. "
        f"Se muestran como máximo {MAX_FRAMES_ANIMACION} generaciones."
    ),
)

usar_semilla = st.sidebar.checkbox(
    "Fijar semilla aleatoria",
    value=True,
    key="usar_semilla",
    help="Desactívala para obtener una trayectoria distinta en cada ejecución.",
)

random_state = st.sidebar.number_input(
    "Semilla aleatoria",
    min_value=0,
    value=42,
    step=1,
    key="semilla",
    disabled=not usar_semilla,
    help="Usa la misma semilla para repetir resultados.",
)

run_button = st.sidebar.button("▶️ Ejecutar simulación", width="stretch")


# -----------------------------------------------------------
# Texto principal
# -----------------------------------------------------------
with st.expander("🧬 Fundamento teórico del algoritmo genético", expanded=False):
    st.markdown(r"""
    Un **algoritmo genético** es un método metaheurístico de optimización inspirado en principios de evolución biológica.  
    Su propósito consiste en aproximar soluciones óptimas mediante la evolución iterativa de una población de candidatos.

    En esta aplicación se considera un problema de **optimización unidimensional**, donde cada individuo de la población
    representa un valor real $ x \in [a,b]$, y el objetivo es aproximar un minimizador de una función $ f(x) $.

    ## Esquema general del método

    ### 1. Inicialización
    Se genera una población inicial de tamaño finito dentro del dominio de búsqueda:
    $$
    x_i^{(0)} \in [a,b], \quad i=1,\dots,N
    $$
    donde $ N $ es el tamaño de la población.

    ### 2. Evaluación
    Cada individuo es evaluado mediante la función objetivo $ f(x) $.  
    Dado que el problema planteado es de **minimización**, los individuos con menor valor de $ f(x) $ se consideran más aptos.

    ### 3. Selección
    Se eligen individuos de la población actual para actuar como progenitores.  
    En esta implementación se emplea **selección por torneo**: se toman $k$ individuos al azar y se elige el de menor $f(x)$, mecanismo que favorece la reproducción de individuos con mejor desempeño relativo.

    ### 4. Crossover
    A partir de dos progenitores, se genera un descendiente mediante combinación lineal convexa:
    $$
    x_{hijo} = \alpha x_1 + (1-\alpha)x_2, \quad \alpha \in [0,1]
    $$
    Este operador permite recombinar información genética de soluciones previamente encontradas.

    ### 5. Mutación
    El descendiente puede experimentar una perturbación aleatoria gaussiana:
    $$
    x' = x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\sigma_t^2)
    $$
    La mutación introduce diversidad en la población y reduce la probabilidad de convergencia prematura hacia óptimos locales.

    La desviación estándar disminuye geométricamente en cada generación $t$:
    $$
    \sigma_t = \sigma_0\,\gamma^{t}, \quad 0 < \gamma \le 1
    $$
    de modo que al inicio se favorece la exploración y al final el refinamiento local.

    ### 6. Elitismo
    El mejor individuo encontrado hasta el momento se preserva en la siguiente generación.  
    Este criterio garantiza que la calidad de la mejor solución histórica no se degrade durante la evolución.

    ### 7. Iteración generacional
    El proceso de selección, recombinación y mutación se repite durante un número prefijado de generaciones,
    produciendo una sucesión de poblaciones:
    $$
    P^{(0)}, P^{(1)}, \dots, P^{(T)}
    $$

    ## Interpretación de los parámetros

    - **Función $$ f(x) $$**  
      Define el problema de optimización. La aplicación busca aproximar un valor de $ x $ que minimice dicha función.

    - **Límite inferior / límite superior del dominio**  
      Delimitan el intervalo de búsqueda:
      $$
      x \in [a,b]
      $$
      Todo individuo generado o mutado es restringido a este intervalo.

    - **Tamaño de población**  
      Número de individuos presentes en cada generación.  
      Un tamaño de población mayor incrementa la diversidad de búsqueda, aunque también eleva el costo computacional.

    - **Cantidad de generaciones**  
      Número de iteraciones evolutivas del algoritmo.  
      En general, un mayor número de generaciones permite una exploración más prolongada del espacio de búsqueda.

    - **Probabilidad de crossover $$ p_c $$**  
      Probabilidad de aplicar recombinación entre dos progenitores.  
      Valores elevados suelen favorecer la explotación de información ya presente en la población.

    - **Probabilidad de mutación $$ p_m $$**  
      Probabilidad de aplicar perturbación aleatoria a un descendiente.  
      Este parámetro regula el grado de exploración estocástica.

    - **Sigma relativo de mutación**  
      Controla la magnitud inicial de la perturbación gaussiana.  
      En la implementación, este valor se escala con la amplitud del dominio:
      $$
      \sigma_0 = \sigma_{relativo}(b-a)
      $$
      Valores pequeños inducen refinamiento local; valores grandes favorecen exploración global.

    - **Enfriamiento de sigma $\gamma$** (parámetros avanzados)  
      Factor por el que se multiplica $\sigma$ en cada generación (por defecto 0.99).  
      Con $\gamma = 1$ la magnitud de la mutación es constante; con valores menores la búsqueda se vuelve más local con el tiempo.

    - **Tamaño del torneo $k$** (parámetros avanzados)  
      Número de individuos que compiten en cada selección (por defecto 3).  
      Un $k$ mayor aumenta la presión selectiva y acelera la convergencia, a costa de diversidad.

    - **Semilla aleatoria**  
      Permite reproducibilidad experimental.  
      Para una misma semilla y los mismos parámetros, la ejecución produce la misma trayectoria evolutiva.  
      Si se desactiva *Fijar semilla aleatoria*, cada ejecución sigue una trayectoria distinta.

    - **Velocidad de animación**  
      Modifica únicamente la representación visual del proceso evolutivo; no altera el comportamiento del algoritmo.  
      Si hay más de 100 generaciones, la animación muestra 100 de ellas espaciadas uniformemente.

    ## Interpretación de la visualización

    - La **curva** representa la función objetivo $ f(x) $.
    - Los **puntos** representan la población en una generación dada.
    - La **estrella** señala el mejor individuo histórico encontrado hasta ese instante.
    - La gráfica de evolución complementaria muestra la sucesión del mejor valor histórico de $ f(x) $ a lo largo de las generaciones.
    - El control **Explorar generación** permite revisar cualquier generación sin repetir la animación; la línea punteada de la gráfica de evolución marca la generación seleccionada.

    ## Observación metodológica
    El algoritmo genético es un método heurístico y estocástico.  
    En consecuencia, no garantiza la obtención exacta del mínimo global en todos los casos; sin embargo,
    constituye una herramienta robusta para aproximar soluciones de alta calidad en problemas donde los métodos analíticos
    o deterministas pueden resultar difíciles de aplicar.
    """)


# -----------------------------------------------------------
# Ejemplos rápidos
# -----------------------------------------------------------
st.subheader("💡 Ejemplos rápidos")

for inicio in range(0, len(EJEMPLOS), 4):
    columnas = st.columns(4)

    for columna, (etiqueta, expr) in zip(columnas, EJEMPLOS[inicio:inicio + 4]):
        with columna:
            st.button(
                etiqueta,
                width="stretch",
                on_click=set_example,
                args=(expr,),
            )

st.code(st.session_state.func, language="python")


# -----------------------------------------------------------
# Ejecución principal
# Los resultados se guardan en session_state: Streamlit vuelve a
# ejecutar el script con cada interacción y, sin esto, desaparecían
# al mover cualquier control.
# -----------------------------------------------------------
if run_button:
    # Validación simple del dominio
    if x_min >= x_max:
        limpiar_resultados()
        st.error("El límite inferior debe ser menor que el límite superior.")
        st.stop()

    try:
        # Construir función desde el string
        f = build_function(eq_str)

        # Muestreo de la curva de fondo
        x_plot = np.linspace(x_min, x_max, 600)
        y_plot = evaluar_vector_seguro(f, x_plot)

        if np.isfinite(y_plot).sum() < 2:
            raise ValueError(
                "La función no pudo evaluarse correctamente en el dominio seleccionado."
            )

        semilla = int(random_state) if usar_semilla else None

        # Crear y ejecutar el algoritmo genético
        ag = GeneticAlgorithm1D(
            f=f,
            bounds=(x_min, x_max),
            pop_size=int(pop_size),
            pc=pc,
            pm=pm,
            sigma=sigma,
            generations=int(generations),
            tournament_k=int(tournament_k),
            sigma_decay=sigma_decay,
            random_state=semilla,
        )

        with st.spinner("Ejecutando algoritmo genético..."):
            best_x, best_f, history, best_history = ag.run()

    except Exception as e:
        limpiar_resultados()
        st.error(f"Error al ejecutar la simulación: {e}")
        st.stop()

    st.session_state.ejecuciones = st.session_state.get("ejecuciones", 0) + 1
    st.session_state.resultado = {
        "id": st.session_state.ejecuciones,
        "expr": eq_str.strip(),
        "bounds": (x_min, x_max),
        "semilla": semilla if semilla is not None else "aleatoria",
        "x_plot": x_plot,
        "y_plot": y_plot,
        "history": history,
        "values_history": ag.values_history,
        "best_history": best_history,
        "best_x": best_x,
        "best_f": best_f,
    }

if "resultado" in st.session_state:
    mostrar_resultados(
        st.session_state.resultado,
        animar=run_button,
        frame_delay=frame_delay,
    )
