import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from AG_procesamiento import (
    MAX_LONGITUD_EXPRESION,
    GeneticAlgorithm1D,
    build_function,
    expresion_latex,
)


# Máximo de cuadros en la animación. Con más generaciones se muestran
# generaciones espaciadas uniformemente (siempre la 0 y la última).
MAX_FRAMES_ANIMACION = 100

# Duración de cada cuadro de la animación (ms)
VELOCIDADES = {"Lenta": 250, "Normal": 100, "Rápida": 40}

# Altura de la vista previa y de la gráfica de la población en la animación (px)
ALTO_GRAFICA = 620

# Colores de las gráficas. Se fijan aquí porque la paleta por defecto pinta la
# población en un azul claro casi igual al de la curva y se perdía sobre ella.
COLOR_CURVA = "#4A6FA5"      # azul medio
COLOR_POBLACION = "#F28C28"  # naranja, con borde oscuro
COLOR_BORDE_POBLACION = "#6B3500"
COLOR_MEJOR = "#7B1FA2"      # morado, con borde blanco

# Ejemplos: cada uno carga su función y un dominio donde se aprecia bien
EJEMPLOS = [
    {
        "nombre": "Combinada",
        "expr": "x**2 + 10*cos(2*x) + x*sin(5*x)",
        "dominio": (-10.0, 10.0),
        "descripcion": "Parábola con oscilaciones: varios mínimos locales alrededor del global.",
    },
    {
        "nombre": "Parábola",
        "expr": "x**2",
        "dominio": (-10.0, 10.0),
        "descripcion": "Un único mínimo en x = 0. El caso más sencillo.",
    },
    {
        "nombre": "Seno",
        "expr": "sin(2*pi*x/3)",
        "dominio": (-6.0, 6.0),
        "descripcion": "Periódica: varios mínimos globales con el mismo valor (−1).",
    },
    {
        "nombre": "Gaussiana",
        "expr": "exp(-x**2)",
        "dominio": (-3.0, 3.0),
        "descripcion": "Campana: al minimizar, la población se va a los bordes del dominio.",
    },
    {
        "nombre": "Valor absoluto",
        "expr": "abs(x)",
        "dominio": (-10.0, 10.0),
        "descripcion": "Mínimo en un pico (sin derivada en x = 0); el AG no la necesita.",
    },
    {
        "nombre": "Doble pozo",
        "expr": "(x**2 - 1)**2",
        "dominio": (-2.0, 2.0),
        "descripcion": "Dos mínimos globales en x = ±1: la población puede repartirse entre ambos.",
    },
    {
        "nombre": "Múltiples mínimos",
        "expr": "x*sin(10*x)",
        "dominio": (-10.0, 10.0),
        "descripcion": "Muchos mínimos locales; el global está cerca de los bordes.",
    },
    {
        "nombre": "Logarítmica",
        "expr": "log(abs(x) + 1)",
        "dominio": (-10.0, 10.0),
        "descripcion": "Mínimo en x = 0 con crecimiento lento: valles poco pronunciados.",
    },
    {
        "nombre": "Sinusoidal",
        "expr": "sin(5*x)",
        "dominio": (-3.0, 3.0),
        "descripcion": "Mínimos globales repetidos de valor −1.",
    },
    {
        "nombre": "Pendiente con oscilación",
        "expr": "0.1*x + sin(3*x)",
        "dominio": (-10.0, 10.0),
        "descripcion": "Oscilación sobre una pendiente: el mínimo global está en el extremo izquierdo.",
    },
    {
        "nombre": "Valles múltiples",
        "expr": "sin(x) + sin(10*x/3)",
        "dominio": (2.7, 7.5),
        "descripcion": "Función de prueba clásica: mínimo global en x ≈ 5.15 entre varios valles.",
    },
    {
        "nombre": "Terreno irregular",
        "expr": "0.3*sin(8*x) + 0.7*cos(15*x)",
        "dominio": (-2.0, 2.0),
        "descripcion": "Muy rugosa: muchos mínimos locales casi tan buenos como el global.",
    },
    {
        "nombre": "Escalón sigmoide",
        "expr": "1/(1 + exp(-10*x))",
        "dominio": (-2.0, 2.0),
        "descripcion": "Meseta: el mínimo está en el borde izquierdo del dominio.",
    },
]
EJEMPLOS_POR_NOMBRE = {ej["nombre"]: ej for ej in EJEMPLOS}
PERSONALIZADA = "— Personalizada —"

AYUDA_SINTAXIS = (
    "Usa la variable `x`, los operadores `+ - * / **` (o `^`), "
    "las funciones `sin cos tan exp log sqrt abs` y la constante `pi`. "
    f"Máximo {MAX_LONGITUD_EXPRESION} caracteres."
)


# -----------------------------------------------------------
# Configuración general de la página
# -----------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Algoritmo Genético 1D",
    page_icon="🧬",
    layout="wide",
)


# -----------------------------------------------------------
# Estado inicial
# Función, ejemplo y dominio se inicializan aquí (y no con value=)
# porque los ejemplos los cambian desde un callback.
# -----------------------------------------------------------
_inicial = EJEMPLOS[0]
st.session_state.setdefault("func", _inicial["expr"])
st.session_state.setdefault("ejemplo", _inicial["nombre"])
st.session_state.setdefault("x_min", _inicial["dominio"][0])
st.session_state.setdefault("x_max", _inicial["dominio"][1])


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


def aplicar_ejemplo():
    """
    Carga la función y el dominio del ejemplo elegido.
    """
    ejemplo = EJEMPLOS_POR_NOMBRE.get(st.session_state.ejemplo)
    if ejemplo is not None:
        st.session_state.func = ejemplo["expr"]
        st.session_state.x_min, st.session_state.x_max = ejemplo["dominio"]


def marcar_personalizada():
    """
    Si el usuario edita la función, el selector deja de mostrar un ejemplo.
    """
    ejemplo = EJEMPLOS_POR_NOMBRE.get(st.session_state.ejemplo)
    if ejemplo is None or ejemplo["expr"] != st.session_state.func.strip():
        st.session_state.ejemplo = PERSONALIZADA


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


@st.cache_data(show_spinner=False, max_entries=64)
def preparar_problema(expr, a, b):
    """
    Valida la función en el dominio y muestrea su curva.
    Lanza ValueError con un mensaje para el usuario si algo no es válido.
    """
    latex = expresion_latex(expr)
    f = build_function(expr)

    x_plot = np.linspace(a, b, 600)
    y_plot = evaluar_vector_seguro(f, x_plot)

    if np.isfinite(y_plot).sum() < 2:
        raise ValueError(
            "La función no tiene valores válidos en este dominio. Prueba con otro intervalo."
        )

    return {"latex": latex, "x_plot": x_plot, "y_plot": y_plot}


def evaluar_problema(expr, a, b):
    """
    Devuelve (problema, error): uno de los dos es None.
    """
    if not expr.strip():
        return None, "Escribe una función f(x) para comenzar."

    if a >= b:
        return None, "El límite inferior debe ser menor que el límite superior."

    try:
        return preparar_problema(expr.strip(), float(a), float(b)), None
    except Exception as e:
        return None, str(e)


def figura_vista_previa(problema, expr, a, b):
    """
    Curva de f(x) antes de ejecutar.
    """
    fig = go.Figure(
        go.Scatter(
            x=problema["x_plot"],
            y=problema["y_plot"],
            mode="lines",
            name=f"f(x) = {expr}",
            line={"width": 2, "color": COLOR_CURVA},
            showlegend=True,
        )
    )
    fig.update_layout(
        title="Vista previa: pulsa Ejecutar para lanzar la población",
        height=ALTO_GRAFICA,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        margin={"t": 70},
    )
    fig.update_xaxes(title_text="x", range=[a, b])
    fig.update_yaxes(title_text="f(x)")
    return fig


def generaciones_animadas(total_gens):
    """
    Generaciones que se incluyen como cuadros de la animación.
    """
    return np.unique(
        np.linspace(0, total_gens, min(total_gens + 1, MAX_FRAMES_ANIMACION)).round().astype(int)
    )


def titulo_generacion(resultado, gen):
    """
    Título con la generación mostrada y el mejor histórico hasta ella.
    """
    total = len(resultado["history"]) - 1
    best_x_hist, best_f_hist = resultado["best_history"][gen]
    return (
        f"Generación {gen}/{total}  ·  "
        f"Mejor histórico: x = {best_x_hist:.6f}, f(x) = {best_f_hist:.6f}"
    )


def trazas_generacion(resultado, gen):
    """
    Trazas que cambian en cada cuadro: población, mejor histórico,
    marcador en la curva de convergencia y texto con la generación.
    El texto es una traza y no el título porque, al animar sin redibujar
    (redraw=False, necesario para que los puntos se desplacen suavemente),
    Plotly no actualiza el título.
    """
    population = resultado["history"][gen]
    pop_y = resultado["values_history"][gen]
    valid_pop = np.isfinite(pop_y)
    best_x_hist, best_f_hist = resultado["best_history"][gen]

    return [
        go.Scatter(x=population[valid_pop], y=pop_y[valid_pop]),
        go.Scatter(x=[best_x_hist], y=[best_f_hist]),
        go.Scatter(x=[gen], y=[best_f_hist]),
        go.Scatter(text=[titulo_generacion(resultado, gen)]),
    ]


def figura_animacion(resultado, duracion_ms):
    """
    Figura de Plotly con la evolución de la población.

    Todas las generaciones se envían al navegador como cuadros (frames) y
    Plotly las reproduce ahí, con sus propios botones y control deslizante.
    Antes la animación era un ciclo en el servidor que mandaba una imagen por
    cuadro: cualquier re-ejecución de Streamlit lo interrumpía y, a través de
    la red, los cuadros llegaban tarde o no se veían.
    """
    total_gens = len(resultado["history"]) - 1
    gens = generaciones_animadas(total_gens)
    x_plot, y_plot = resultado["x_plot"], resultado["y_plot"]
    best_values = [best_f_hist for _, best_f_hist in resultado["best_history"]]

    # Rango del eje Y: curva + todos los valores válidos de la población
    all_y_values = np.concatenate(
        [y_plot[np.isfinite(y_plot)]]
        + [values[np.isfinite(values)] for values in resultado["values_history"]]
    )
    y_min, y_max = float(all_y_values.min()), float(all_y_values.max())
    y_margen = 0.1 * max(y_max - y_min, 1e-6)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.13,
        subplot_titles=("", "Evolución del mejor valor encontrado"),
    )

    # Trazas fijas (0 y 3) y trazas que cambian en cada cuadro (1, 2, 4 y 5)
    poblacion, mejor, marcador, texto = trazas_generacion(resultado, 0)
    fig.add_trace(
        go.Scatter(x=x_plot, y=y_plot, mode="lines", name=f"f(x) = {resultado['expr']}",
                   line={"width": 2, "color": COLOR_CURVA}, hoverinfo="skip"),
        row=1, col=1,
    )
    poblacion.update(mode="markers", name="Población",
                     marker={"size": 11, "color": COLOR_POBLACION, "opacity": 0.9,
                             "line": {"width": 1.5, "color": COLOR_BORDE_POBLACION}},
                     hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra>Individuo</extra>")
    fig.add_trace(poblacion, row=1, col=1)
    mejor.update(mode="markers", name="Mejor histórico",
                 marker={"size": 22, "symbol": "star", "color": COLOR_MEJOR,
                         "line": {"width": 1.5, "color": "white"}},
                 hovertemplate="x = %{x:.6f}<br>f(x) = %{y:.6f}<extra>Mejor histórico</extra>")
    fig.add_trace(mejor, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=list(range(total_gens + 1)), y=best_values, mode="lines+markers",
                   name="Mejor f(x) histórico", line={"color": COLOR_CURVA},
                   marker={"size": 4, "color": COLOR_CURVA}, showlegend=False,
                   hovertemplate="Generación %{x}<br>f(x) = %{y:.6f}<extra></extra>"),
        row=2, col=1,
    )
    marcador.update(mode="markers",
                    marker={"size": 13, "symbol": "circle-open", "color": COLOR_MEJOR,
                            "line": {"width": 3, "color": COLOR_MEJOR}},
                    showlegend=False, hoverinfo="skip")
    fig.add_trace(marcador, row=2, col=1)
    texto.update(x=[resultado["bounds"][0]], y=[y_max + y_margen], mode="text",
                 textposition="bottom right", textfont={"size": 14}, showlegend=False, hoverinfo="skip")
    fig.add_trace(texto, row=1, col=1)

    fig.frames = [
        go.Frame(
            name=str(gen),
            data=trazas_generacion(resultado, gen),
            traces=[1, 2, 4, 5],
        )
        for gen in gens
    ]

    cuadro_inmediato = {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                        "transition": {"duration": 0}}
    reproducir = {
        "frame": {"duration": duracion_ms, "redraw": False},
        "fromcurrent": True,
        "mode": "immediate",
        "transition": {"duration": int(duracion_ms * 0.7), "easing": "cubic-in-out"},
    }

    fig.update_layout(
        # Identifica la figura de cada ejecución (lo usa _JS_REPRODUCIR)
        meta={"ejecucion": resultado["id"]},
        height=ALTO_GRAFICA + 320,
        margin={"t": 100, "b": 130},
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0,
            "xanchor": "left",
            "y": 1.09,
            "yanchor": "bottom",
            "pad": {"r": 6},
            "showactive": False,
            "buttons": [
                {"label": "▶ Reproducir", "method": "animate", "args": [None, reproducir]},
                {"label": "⏸ Pausa", "method": "animate", "args": [[None], cuadro_inmediato]},
                {"label": "⏮ Inicio", "method": "animate", "args": [[str(gens[0])], cuadro_inmediato]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0,
            "len": 1,
            "y": 0,
            "yanchor": "top",
            "pad": {"t": 70},
            "currentvalue": {"prefix": "Generación: "},
            "steps": [
                {"label": str(gen), "method": "animate", "args": [[str(gen)], cuadro_inmediato]}
                for gen in gens
            ],
        }],
    )
    fig.update_xaxes(title_text="x", range=list(resultado["bounds"]), row=1, col=1)
    fig.update_yaxes(title_text="f(x)", range=[y_min - y_margen, y_max + y_margen], row=1, col=1)
    fig.update_xaxes(title_text="Generación", range=[-0.5, total_gens + 0.5], row=2, col=1)
    fig.update_yaxes(title_text="Mejor f(x)", row=2, col=1)
    return fig


# Pulsa "▶ Reproducir" de la animación en cuanto Plotly termina de dibujarla.
# Plotly no tiene una opción para que una animación arranque sola; así basta
# con pulsar Ejecutar simulación. El script es fijo (solo se le inserta el id
# numérico de la ejecución) y se envía únicamente justo después de ejecutar,
# así que mover otros controles no vuelve a reproducir la animación.
_JS_REPRODUCIR = """
<script>
(() => {
  const ejecucion = %d;
  if (window.__agAnimacionReproducida === ejecucion) return;
  const inicio = Date.now();
  const intentar = () => {
    // Solo la figura de esta ejecución: mientras se dibuja la nueva, la de la
    // ejecución anterior sigue un momento en la página.
    const graficas = [...document.querySelectorAll(".js-plotly-plot")];
    const grafica = graficas.find(
      (g) => g.layout && g.layout.meta && g.layout.meta.ejecucion === ejecucion
    );
    const boton = grafica && [...grafica.querySelectorAll(".updatemenu-button")].find(
      (b) => b.textContent.includes("Reproducir")
    );
    if (boton) {
      window.__agAnimacionReproducida = ejecucion;
      boton.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    } else if (Date.now() - inicio < 15000) {
      setTimeout(intentar, 100);
    }
  };
  intentar();
})();
</script>
"""


def reproducir_al_cargar(id_ejecucion):
    """
    Inicia la animación automáticamente (ver _JS_REPRODUCIR).
    """
    st.html(_JS_REPRODUCIR % int(id_ejecucion), unsafe_allow_javascript=True)


def mostrar_resultados(resultado, recien_ejecutado, duracion_ms):
    """
    Dibuja los resultados guardados en session_state: la animación
    (se reproduce en el navegador), las métricas y el botón para limpiar.
    """
    total_gens = len(resultado["history"]) - 1
    a, b = resultado["bounds"]

    if recien_ejecutado:
        st.success("Simulación completada.", icon="✅")

    st.plotly_chart(
        figura_animacion(resultado, duracion_ms),
        width="stretch",
        key=f"animacion_{resultado['id']}",
        config={"displaylogo": False},
    )

    if recien_ejecutado:
        reproducir_al_cargar(resultado["id"])

    st.caption(
        f"Resultados para f(x) = `{resultado['expr']}` en [{a}, {b}]"
        f"  ·  semilla: {resultado['semilla']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mejor x encontrado", format_value(resultado["best_x"]))

    with col2:
        st.metric("Mejor f(x)", format_value(resultado["best_f"]))

    with col3:
        st.metric("Generaciones evaluadas", str(total_gens))

    st.button("🗑️ Limpiar resultados", width="stretch", on_click=limpiar_resultados)


# -----------------------------------------------------------
# Barra lateral: parámetros del algoritmo y de la visualización
# -----------------------------------------------------------
st.sidebar.header("⚙️ Parámetros")

st.sidebar.subheader("🧬 Algoritmo")

pop_size = st.sidebar.number_input(
    "Tamaño de población",
    min_value=4,
    max_value=500,
    value=50,
    step=1,
    key="pop_size",
    help="Cantidad de individuos (soluciones candidatas) en cada generación.",
)

generations = st.sidebar.number_input(
    "Cantidad de generaciones",
    min_value=1,
    max_value=500,
    value=80,
    step=1,
    key="generations",
    help="Cuántas veces se repite el ciclo de selección, cruce y mutación.",
)

pc = st.sidebar.slider(
    "Probabilidad de crossover (pc)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
    key="pc",
    help="Probabilidad de combinar dos padres para crear un hijo. "
    "Valores altos aprovechan más las buenas soluciones ya encontradas.",
)

pm = st.sidebar.slider(
    "Probabilidad de mutación (pm)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    key="pm",
    help="Probabilidad de alterar al azar a un hijo. Más mutación = más exploración.",
)

sigma = st.sidebar.number_input(
    "Sigma relativo de mutación",
    min_value=0.001,
    max_value=1.0,
    value=0.10,
    step=0.01,
    key="sigma",
    help="Qué tanto se mueve un individuo al mutar, como fracción del ancho del dominio "
    "(0.10 = 10 % del rango).",
)

with st.sidebar.expander("🔧 Avanzado"):
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

    usar_semilla = st.checkbox(
        "Fijar semilla aleatoria",
        value=True,
        key="usar_semilla",
        help="Desactívala para obtener una trayectoria distinta en cada ejecución.",
    )

    random_state = st.number_input(
        "Semilla aleatoria",
        min_value=0,
        value=42,
        step=1,
        key="semilla",
        disabled=not usar_semilla,
        help="Con la misma semilla y los mismos parámetros, el resultado se repite.",
    )

st.sidebar.subheader("🎞️ Visualización")

velocidad = st.sidebar.radio(
    "Velocidad de la animación",
    list(VELOCIDADES),
    index=1,
    horizontal=True,
    key="velocidad",
    help="Velocidad de la animación (al ejecutar y con el botón ▶ Reproducir). "
    f"Se muestran como máximo {MAX_FRAMES_ANIMACION} generaciones.",
)


# -----------------------------------------------------------
# Encabezado y definición del problema
# -----------------------------------------------------------
st.title("Simulador de Algoritmo Genético 1D")
st.caption(
    "Busca el mínimo de una función f(x) con un algoritmo genético "
    "y observa cómo evoluciona la población, generación por generación."
)

col_func, col_ejemplo = st.columns([3, 2])

with col_func:
    eq_str = st.text_input(
        "Función f(x) a minimizar",
        key="func",
        placeholder="Ej.: x**2 + 10*cos(2*x)",
        help=AYUDA_SINTAXIS,
        on_change=marcar_personalizada,
    )

with col_ejemplo:
    st.selectbox(
        "Ejemplo",
        [PERSONALIZADA] + list(EJEMPLOS_POR_NOMBRE),
        key="ejemplo",
        on_change=aplicar_ejemplo,
        help="Carga una función junto con un dominio donde se aprecia bien.",
    )
    ejemplo_actual = EJEMPLOS_POR_NOMBRE.get(st.session_state.ejemplo)
    if ejemplo_actual is not None:
        st.caption(ejemplo_actual["descripcion"])

col_a, col_b, col_ejecutar = st.columns(3, vertical_alignment="bottom")

with col_a:
    x_min = st.number_input("Límite inferior (a)", key="x_min")

with col_b:
    x_max = st.number_input("Límite superior (b)", key="x_max")

problema, error_problema = evaluar_problema(eq_str, x_min, x_max)

# El botón no se deshabilita cuando hay un error: al corregir la función y hacer clic
# directamente, el navegador confirma el texto y registra el clic con el botón todavía
# deshabilitado, así que el clic se perdía y había que pulsar dos veces.
with col_ejecutar:
    run_button = st.button(
        "▶️ Ejecutar simulación",
        type="primary",
        width="stretch",
        key="ejecutar",
    )

# Validación en vivo, junto a los campos: fórmula interpretada o error
if problema is not None:
    st.latex(f"f(x) = {problema['latex']}")
else:
    st.error(error_problema)

zona_mensajes = st.container()

if run_button and problema is None:
    zona_mensajes.warning("Corrige el error indicado arriba para ejecutar la simulación.", icon="⚠️")


# -----------------------------------------------------------
# Ejecución principal
# Los resultados se guardan en session_state: Streamlit vuelve a
# ejecutar el script con cada interacción y, sin esto, desaparecían
# al mover cualquier control.
# -----------------------------------------------------------
problema_actual = (eq_str.strip(), float(x_min), float(x_max))
parametros_actuales = {
    "pop_size": int(pop_size),
    "generations": int(generations),
    "pc": pc,
    "pm": pm,
    "sigma": sigma,
    "tournament_k": int(tournament_k),
    "sigma_decay": sigma_decay,
    "semilla": int(random_state) if usar_semilla else None,
}

recien_ejecutado = False

if run_button and problema is not None:
    semilla = parametros_actuales["semilla"]

    try:
        ag = GeneticAlgorithm1D(
            f=build_function(eq_str),
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
        zona_mensajes.error(f"No se pudo ejecutar la simulación: {e}")

    else:
        st.session_state.ejecuciones = st.session_state.get("ejecuciones", 0) + 1
        st.session_state.resultado = {
            "id": st.session_state.ejecuciones,
            "expr": eq_str.strip(),
            "bounds": (x_min, x_max),
            "semilla": semilla if semilla is not None else "aleatoria",
            "x_plot": problema["x_plot"],
            "y_plot": problema["y_plot"],
            "history": history,
            "values_history": ag.values_history,
            "best_history": best_history,
            "best_x": best_x,
            "best_f": best_f,
            "problema": problema_actual,
            "parametros": parametros_actuales,
        }
        recien_ejecutado = True

resultado = st.session_state.get("resultado")

if resultado is not None and resultado["problema"] == problema_actual:
    # Mismo problema: se muestran los resultados, avisando si cambió algún parámetro
    if resultado["parametros"] != parametros_actuales:
        zona_mensajes.info(
            "Cambiaste parámetros del algoritmo desde la última ejecución. "
            "Pulsa **Ejecutar simulación** para actualizar los resultados.",
            icon="🔄",
        )

    mostrar_resultados(
        resultado,
        recien_ejecutado=recien_ejecutado,
        duracion_ms=VELOCIDADES[velocidad],
    )

else:
    # Problema nuevo (o sin ejecutar): vista previa de la función
    if resultado is not None:
        zona_mensajes.info(
            "La función o el dominio cambiaron. Pulsa **Ejecutar simulación** "
            "para simular este problema.",
            icon="🔄",
        )

    if problema is not None:
        st.plotly_chart(
            figura_vista_previa(problema, *problema_actual),
            width="stretch",
            config={"displaylogo": False},
        )


# -----------------------------------------------------------
# Ayuda y fundamento teórico (al final, para no competir con la herramienta)
# -----------------------------------------------------------
st.divider()

with st.expander("ℹ️ Cómo usar el simulador", expanded=False):
    st.markdown(
        f"""
        1. **Elige un ejemplo** o escribe tu propia función en *Función f(x) a minimizar*.
           Los ejemplos cargan también un dominio donde se aprecia bien.
        2. **Ajusta el dominio** [a, b]. La fórmula interpretada aparece debajo; si hay un
           error, se indica ahí mismo.
        3. Revisa la **vista previa** de la curva y pulsa **▶️ Ejecutar simulación**.
        4. La animación arranca sola. **⏸ Pausa** la detiene, **▶ Reproducir** la continúa
           y **⏮ Inicio** vuelve a la generación 0.
        5. Ajusta los parámetros del algoritmo y la velocidad de la animación en la **barra lateral**.

        **Sintaxis:** {AYUDA_SINTAXIS}

        **Interpretación visual:**
        - Los puntos representan la población.
        - La estrella representa el mejor valor histórico encontrado.
        - La generación 0 es la población inicial.
        - El control *Generación* debajo de la gráfica permite revisar cualquier generación;
          el círculo en la gráfica de convergencia marca la generación mostrada.
        - Al pasar el mouse sobre un punto se ven sus valores de x y f(x).
        - Los resultados se conservan hasta la siguiente simulación o hasta pulsar *Limpiar resultados*.
          Si cambias un parámetro, aparece un aviso para volver a ejecutar.
        """
    )

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

    - **Velocidad de la animación** (barra lateral → Visualización)
      Define la rapidez de la animación: lenta, normal o rápida.
      Solo modifica la representación visual del proceso evolutivo; no altera el comportamiento del algoritmo.
      Si hay más de 100 generaciones, la animación muestra 100 de ellas espaciadas uniformemente.

    ## Interpretación de la visualización

    - La **curva** representa la función objetivo $ f(x) $.
    - Los **puntos** representan la población en una generación dada.
    - La **estrella** señala el mejor individuo histórico encontrado hasta ese instante.
    - La gráfica de evolución complementaria muestra la sucesión del mejor valor histórico de $ f(x) $ a lo largo de las generaciones.
    - La animación arranca sola al ejecutar y se reproduce en el navegador; los botones **▶ Reproducir**, **⏸ Pausa** y **⏮ Inicio** la controlan.
    - El control **Generación** permite revisar cualquier generación; el círculo de la gráfica de evolución marca la generación mostrada.

    ## Observación metodológica
    El algoritmo genético es un método heurístico y estocástico.  
    En consecuencia, no garantiza la obtención exacta del mínimo global en todos los casos; sin embargo,
    constituye una herramienta robusta para aproximar soluciones de alta calidad en problemas donde los métodos analíticos
    o deterministas pueden resultar difíciles de aplicar.
    """)
