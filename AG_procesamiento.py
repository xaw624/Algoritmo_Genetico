import ast
import math
import operator

import numpy as np
import sympy as sp


# -----------------------------------------------------------
# Validación de expresiones
# sympify usa eval internamente, así que la expresión se revisa
# con el AST de Python ANTES de pasársela. Solo se aceptan números,
# x, pi, operadores aritméticos y las funciones de la lista.
# -----------------------------------------------------------
MAX_LONGITUD_EXPRESION = 200

_FUNCIONES_PERMITIDAS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "Abs": abs,
}
_NOMBRES_PERMITIDOS = {"x", "pi"} | set(_FUNCIONES_PERMITIDAS)

_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_NODOS_PERMITIDOS = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.USub, ast.UAdd, *_OPERADORES,
)


def _normalizar(expr_str):
    """
    Convierte ^ en ** como hace sympify (convert_xor).
    Hay que hacerlo ANTES de analizar el AST: en Python 9^9^9 se agrupa
    como (9^9)^9, pero sympify lo convierte en 9**(9**9).
    """
    return expr_str.strip().replace("^", "**")


def _valor_constante(node):
    """
    Evalúa con floats un subárbol que no depende de x.
    Devuelve None si depende de x o no tiene valor real (p. ej. log(-1)).

    Lanza ValueError si alguna constante desborda un float. Así se evita
    que SymPy o Python intenten calcular enteros gigantes (9**9**9)
    y bloqueen el servidor.
    """
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return math.pi if node.id == "pi" else None

    if isinstance(node, ast.UnaryOp):
        operandos = [node.operand]
    elif isinstance(node, ast.BinOp):
        operandos = [node.left, node.right]
    elif isinstance(node, ast.Call):
        operandos = node.args
    else:
        return None

    # Se evalúan todos los operandos para revisar cada subárbol constante,
    # aunque el nodo actual dependa de x
    valores = [_valor_constante(n) for n in operandos]
    if any(v is None for v in valores):
        return None

    try:
        if isinstance(node, ast.UnaryOp):
            valor = -valores[0] if isinstance(node.op, ast.USub) else valores[0]
        elif isinstance(node, ast.BinOp):
            valor = _OPERADORES[type(node.op)](*valores)
        else:
            valor = _FUNCIONES_PERMITIDAS[node.func.id](*valores)
    except OverflowError:
        raise ValueError("La expresión contiene una constante demasiado grande.")
    except (ValueError, ZeroDivisionError, TypeError):
        return None

    if isinstance(valor, complex):
        return None
    if not math.isfinite(valor):
        raise ValueError("La expresión contiene una constante demasiado grande.")

    return valor


def validar_expresion(expr_str):
    """
    Verifica que la expresión solo use elementos permitidos.
    Lanza ValueError con un mensaje claro si no es así.
    """
    if len(expr_str) > MAX_LONGITUD_EXPRESION:
        raise ValueError(
            f"La expresión excede {MAX_LONGITUD_EXPRESION} caracteres."
        )

    try:
        tree = ast.parse(_normalizar(expr_str), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Sintaxis inválida: {e.msg}")

    for node in ast.walk(tree):
        if not isinstance(node, _NODOS_PERMITIDOS):
            raise ValueError(f"Elemento no permitido: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id not in _NOMBRES_PERMITIDOS:
            raise ValueError(f"Nombre no permitido: '{node.id}'. Usa solo 'x'.")

        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in _FUNCIONES_PERMITIDAS
        ):
            raise ValueError(
                "Solo se permiten llamadas a sin, cos, tan, exp, log, sqrt y abs."
            )

        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("Solo se permiten constantes numéricas.")

    _valor_constante(tree.body)


def _scalar_to_float_or_inf(value):
    """
    Convierte un resultado cualquiera a float.
    Si no se puede convertir o no es finito, devuelve +inf.
    Esto evita que el AG se rompa con NaN, inf o errores de dominio.
    """
    try:
        arr = np.asarray(value, dtype=float)

        # Nos aseguramos de que sea un único escalar
        if arr.size != 1:
            return np.inf

        scalar = float(arr.reshape(-1)[0])

        # Si no es finito, se castiga como infinito
        if not np.isfinite(scalar):
            return np.inf

        return scalar
    except Exception:
        return np.inf


class GeneticAlgorithm1D:
    """
    Algoritmo Genético 1D para minimizar una función f(x)
    en un dominio continuo [a, b].
    """

    def __init__(
        self,
        f,
        bounds: tuple,
        pop_size=30,
        pc=0.8,
        pm=0.1,
        sigma=0.1,
        generations=200,
        tournament_k=3,
        sigma_decay=0.99,
        random_state=None,
    ):
        """
        Parámetros:
        - f: función objetivo a minimizar
        - bounds: tupla (a, b) con el dominio de búsqueda
        - pop_size: tamaño de la población
        - pc: probabilidad de crossover
        - pm: probabilidad de mutación
        - sigma: tamaño relativo de la mutación respecto al dominio
        - generations: cantidad de generaciones
        - tournament_k: tamaño del torneo para selección
        - sigma_decay: factor de enfriamiento de sigma por generación
        - random_state: semilla opcional para reproducibilidad
        """

        # Función objetivo
        self.f = f

        # Límites del dominio
        self.a, self.b = bounds

        # Validación básica del dominio
        if self.a >= self.b:
            raise ValueError("El límite inferior debe ser menor que el superior.")

        # Validación del tamaño de población
        if pop_size < 2:
            raise ValueError("La población debe tener al menos 2 individuos.")

        # Parámetros principales del algoritmo
        self.pop_size = int(pop_size)
        self.pc = float(pc)
        self.pm = float(pm)
        self.generations = int(generations)

        # Selección por torneo
        self.tournament_k = int(max(2, min(tournament_k, self.pop_size)))

        # Sigma relativo al tamaño del dominio
        domain_size = self.b - self.a
        self.initial_sigma = float(sigma) * domain_size
        self.sigma = self.initial_sigma
        self.sigma_decay = float(sigma_decay)

        # Generador aleatorio moderno de NumPy
        self.rng = np.random.default_rng(random_state)

        # Estado interno
        self.population = None
        self.fitness = None
        self.best_x = None
        self.best_f = np.inf

        # Historial para visualización
        self.history = []         # Guarda la población de cada generación
        self.values_history = []  # Guarda f(x) de cada población (+inf si no es válido)
        self.best_history = []    # Guarda (best_x, best_f) por generación

    def initialize_population(self):
        """
        Crea la población inicial uniforme dentro del dominio [a, b].
        """
        self.population = self.rng.uniform(self.a, self.b, self.pop_size)

    def evaluate_individual(self, x):
        """
        Evalúa un solo individuo de forma segura.
        Si la función falla o devuelve algo inválido, se castiga con +inf.
        """
        try:
            with np.errstate(all="ignore"):
                value = self.f(float(x))
            return _scalar_to_float_or_inf(value)
        except Exception:
            return np.inf

    def evaluate(self):
        """
        Evalúa toda la población actual:
        - Calcula f(x) para cada individuo
        - Convierte minimización a fitness de maximización usando -f(x)
        - Actualiza el mejor individuo histórico
        - Guarda historial para animación
        """
        values = np.array([self.evaluate_individual(x) for x in self.population], dtype=float) # type: ignore

        # Fitness: como el AG selecciona "mayor fitness", usamos -f(x)
        self.fitness = -values

        # Mejor individuo de la generación actual
        idx = np.argmin(values)
        gen_best_x = float(self.population[idx]) # type: ignore
        gen_best_f = float(values[idx])

        # Actualizar mejor histórico
        if gen_best_f < self.best_f:
            self.best_f = gen_best_f
            self.best_x = gen_best_x

        # Guardar historial
        self.history.append(self.population.copy()) # type: ignore
        self.values_history.append(values)
        self.best_history.append((self.best_x, self.best_f))

    def select_parent(self):
        """
        Selección por torneo.
        Elige k individuos al azar y retorna el mejor.
        """
        idx = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        best_local_idx = idx[np.argmax(self.fitness[idx])] # type: ignore
        return float(self.population[best_local_idx]) # type: ignore

    def crossover(self, p1, p2):
        """
        Crossover para variables reales.
        Genera un hijo mediante combinación convexa de dos padres.
        """
        if self.rng.random() < self.pc:
            alpha = self.rng.random()
            return alpha * p1 + (1 - alpha) * p2

        # Si no hay crossover, uno de los padres pasa directo
        return p1

    def mutate(self, x):
        """
        Mutación gaussiana con recorte al dominio.
        """
        if self.rng.random() < self.pm:
            x = x + self.rng.normal(0, self.sigma)

        # Mantener siempre al individuo dentro del dominio
        return float(np.clip(x, self.a, self.b))

    def step(self):
        """
        Ejecuta una generación:
        - Mantiene elitismo: el mejor histórico pasa directo
        - Genera el resto con selección, crossover y mutación
        """
        if self.best_x is None:
            raise RuntimeError("Debes evaluar la población antes de llamar a step().")

        # Elitismo: conservar al mejor histórico
        new_population = [self.best_x]

        # Completar la nueva población
        while len(new_population) < self.pop_size:
            p1 = self.select_parent()
            p2 = self.select_parent()

            child = self.crossover(p1, p2)
            child = self.mutate(child)

            new_population.append(child)

        self.population = np.array(new_population, dtype=float)

    def run(self):
        """
        Ejecuta el algoritmo completo y devuelve:
        - best_x: mejor x encontrado
        - best_f: mejor valor f(x)
        - history: historial de poblaciones
        - best_history: historial del mejor histórico por generación

        Nota:
        Se guarda también la generación 0 (población inicial).
        Los valores f(x) de cada generación quedan en self.values_history.

        Lanza ValueError si ningún individuo de la población inicial
        tiene un valor finito de f(x).
        """
        # Reiniciar estado por si el objeto se reutiliza
        self.population = None
        self.fitness = None
        self.best_x = None
        self.best_f = np.inf
        self.history = []
        self.values_history = []
        self.best_history = []
        self.sigma = self.initial_sigma

        # 1) Población inicial
        self.initialize_population()

        # 2) Evaluar generación 0
        self.evaluate()

        if self.best_x is None:
            raise ValueError(
                "La función no tiene valores finitos en ningún individuo de la "
                "población inicial. Ajusta el dominio o aumenta la población."
            )

        # 3) Evolucionar
        for _ in range(self.generations):
            self.step()
            self.sigma *= self.sigma_decay
            self.evaluate()

        return self.best_x, self.best_f, self.history, self.best_history


def interpretar_expresion(expr_str):
    """
    Valida el texto y lo convierte en una expresión de SymPy en la variable x.
    Devuelve (expr, x). Lanza ValueError con un mensaje claro si no es válida.

    Funciones permitidas:
    sin, cos, tan, exp, log, sqrt, abs, pi
    Variable permitida:
    x

    La expresión se valida antes con validar_expresion() y se interpreta
    con evaluate=False para que SymPy no simplifique términos como
    x - x y termine calculando constantes enormes.
    """
    validar_expresion(expr_str)

    x = sp.symbols("x")

    allowed_locals = {
        "x": x,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "pi": sp.pi,
    }

    try:
        expr = sp.sympify(_normalizar(expr_str), locals=allowed_locals, evaluate=False)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la expresión: {e}")

    # Validar que solo exista la variable x
    invalid_symbols = expr.free_symbols - {x}
    if invalid_symbols:
        raise ValueError(
            f"La expresión contiene variables no permitidas: {invalid_symbols}. Usa solo 'x'."
        )

    return expr, x


# Precedencias para decidir paréntesis en LaTeX
_PREC_SUMA, _PREC_PRODUCTO, _PREC_UNARIO, _PREC_POTENCIA, _PREC_ATOMO = 1, 2, 3, 4, 5

_LATEX_FUNCIONES = {"sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "log": r"\log"}


def _parentesis(texto):
    return rf"\left({texto}\right)"


def _latex_constante(valor):
    texto = repr(valor) if isinstance(valor, float) else str(valor)
    if "e" in texto:
        mantisa, exponente = texto.split("e")
        return rf"{mantisa} \times 10^{{{int(exponente)}}}"
    return texto


def _latex_nodo(node):
    """
    Convierte un nodo del AST (ya validado) en LaTeX.
    Devuelve (latex, precedencia).
    """
    if isinstance(node, ast.Constant):
        return _latex_constante(node.value), _PREC_ATOMO

    if isinstance(node, ast.Name):
        return (r"\pi" if node.id == "pi" else node.id), _PREC_ATOMO

    if isinstance(node, ast.UnaryOp):
        operando, prec = _latex_nodo(node.operand)
        if prec < _PREC_UNARIO:
            operando = _parentesis(operando)
        signo = "-" if isinstance(node.op, ast.USub) else "+"
        return signo + operando, _PREC_UNARIO

    if isinstance(node, ast.Call):
        nombre = node.func.id
        argumento = ", ".join(_latex_nodo(a)[0] for a in node.args)
        if nombre in ("abs", "Abs"):
            return rf"\left|{argumento}\right|", _PREC_ATOMO
        if nombre == "sqrt":
            return rf"\sqrt{{{argumento}}}", _PREC_ATOMO
        if nombre == "exp":
            return rf"e^{{{argumento}}}", _PREC_POTENCIA
        return _LATEX_FUNCIONES[nombre] + _parentesis(argumento), _PREC_ATOMO

    # BinOp
    izq, p_izq = _latex_nodo(node.left)
    der, p_der = _latex_nodo(node.right)

    if isinstance(node.op, ast.Div):
        return rf"\frac{{{izq}}}{{{der}}}", _PREC_ATOMO

    if isinstance(node.op, ast.Pow):
        if p_izq < _PREC_ATOMO:
            izq = _parentesis(izq)
        return f"{izq}^{{{der}}}", _PREC_POTENCIA

    if isinstance(node.op, ast.Mult):
        if p_izq < _PREC_PRODUCTO:
            izq = _parentesis(izq)
        if p_der < _PREC_PRODUCTO or p_der == _PREC_UNARIO:
            der = _parentesis(der)
        # Entre números se usa \cdot; con variables o funciones basta yuxtaponer
        separador = r" \cdot " if isinstance(node.right, ast.Constant) else " "
        return f"{izq}{separador}{der}", _PREC_PRODUCTO

    # Suma y resta: el lado derecho necesita paréntesis si es otra suma
    # (solo en la resta) o un signo unario
    if p_der == _PREC_UNARIO or (isinstance(node.op, ast.Sub) and p_der <= _PREC_SUMA):
        der = _parentesis(der)
    operador = "+" if isinstance(node.op, ast.Add) else "-"
    return f"{izq} {operador} {der}", _PREC_SUMA


def expresion_latex(expr_str):
    """
    Devuelve la expresión en LaTeX tal como la escribió el usuario
    (sin reordenar ni simplificar). Valida antes con interpretar_expresion().
    """
    interpretar_expresion(expr_str)
    tree = ast.parse(_normalizar(expr_str), mode="eval")
    return _latex_nodo(tree.body)[0]


def build_function(expr_str):
    """
    Construye una función numérica f(x) a partir de un string usando SymPy.
    Ver interpretar_expresion() para la sintaxis permitida.
    """
    expr, x = interpretar_expresion(expr_str)

    try:
        raw_f = sp.lambdify(x, expr, modules=["numpy"])
    except Exception as e:
        raise ValueError(f"No se pudo convertir la expresión a función numérica: {e}")

    def f(x_value):
        with np.errstate(all="ignore"):
            return raw_f(x_value)

    return f


# Bloque de prueba local.
# NO se ejecuta cuando este archivo se importa desde app.py
if __name__ == "__main__":
    expr_str = "x**2 + 10*cos(2*x) + x*sin(5*x)"
    f = build_function(expr_str)

    ag = GeneticAlgorithm1D(
        f=f,
        bounds=(-10, 10),
        pop_size=50,
        pc=0.8,
        pm=0.1,
        sigma=0.1,
        generations=100,
        random_state=42,
    )

    best_x, best_f, history, best_history = ag.run()
    print(f"Mejor x: {best_x}")
    print(f"Mejor f(x): {best_f}")
    print(f"Generaciones registradas: {len(history)}")

