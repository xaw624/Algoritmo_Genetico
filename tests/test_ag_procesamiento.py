import time

import numpy as np
import pytest

from AG_procesamiento import (
    MAX_LONGITUD_EXPRESION,
    GeneticAlgorithm1D,
    build_function,
    expresion_latex,
    validar_expresion,
)

# Mismas expresiones que los ejemplos rápidos de app.py
EJEMPLOS = [
    "x**2",
    "sin(2*pi*x/3)",
    "exp(-x**2)",
    "abs(x)",
    "(x**2 - 1)**2",
    "x*sin(10*x)",
    "log(abs(x) + 1)",
    "sin(5*x)",
    "0.1*x + sin(3*x)",
    "sin(x) + sin(10*x/3)",
    "0.3*sin(8*x) + 0.7*cos(15*x)",
    "1/(1 + exp(-10*x))",
]


def correr(expr, bounds=(-10, 10), **kwargs):
    params = dict(pop_size=50, generations=80, random_state=42)
    params.update(kwargs)
    ag = GeneticAlgorithm1D(build_function(expr), bounds, **params)
    return ag, ag.run()


# -----------------------------------------------------------
# build_function / validar_expresion
# -----------------------------------------------------------
@pytest.mark.parametrize("expr", EJEMPLOS + ["x^2 + 1", "  x**(1/3)", "-x + 3.5e-2", "Abs(x)"])
def test_expresiones_validas(expr):
    f = build_function(expr)
    assert np.isfinite(f(0.5))


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hola')",
        "().__class__.__bases__",
        "x.real",
        "lambda: 1",
        "open('archivo')",
        "y + x",
        "sin(x, evaluate=False)",
        "[x]",
        "True * x",
        "1j * x",
        "'texto'",
        "x**2 - 1)**2",
        "x" * (MAX_LONGITUD_EXPRESION + 1),
    ],
)
def test_expresiones_rechazadas(expr):
    with pytest.raises(ValueError):
        build_function(expr)


@pytest.mark.parametrize("expr", ["9**9**9", "(9**100)**100", "9**abs(9**9)", "exp(1000) * x"])
def test_constantes_enormes_se_rechazan(expr):
    with pytest.raises(ValueError, match="demasiado grande"):
        validar_expresion(expr)


def test_no_se_simplifica_hacia_constantes_enormes():
    # Con evaluate=True, SymPy simplificaba x - x y calculaba 9**387420489
    inicio = time.perf_counter()
    f = build_function("9**(x - x + 9**9)")
    ag = GeneticAlgorithm1D(f, (-1, 1), pop_size=4, generations=1)
    assert ag.evaluate_individual(0.5) == np.inf
    assert time.perf_counter() - inicio < 5


# -----------------------------------------------------------
# GeneticAlgorithm1D
# -----------------------------------------------------------
@pytest.mark.parametrize("expr", EJEMPLOS)
def test_encuentra_el_minimo_de_los_ejemplos(expr):
    f = build_function(expr)
    xs = np.linspace(-10, 10, 200_001)
    with np.errstate(all="ignore"):
        minimo_real = np.nanmin(f(xs))

    _, (best_x, best_f, _, _) = correr(expr)

    assert -10 <= best_x <= 10
    assert best_f == pytest.approx(minimo_real, abs=1e-2)


def test_historiales_alineados_y_dentro_del_dominio():
    ag, (_, _, history, best_history) = correr("x**2", bounds=(-3, 5), generations=30)

    assert len(history) == len(ag.values_history) == len(best_history) == 31
    for population, values in zip(history, ag.values_history):
        assert population.shape == values.shape == (50,)
        assert np.all((population >= -3) & (population <= 5))
        np.testing.assert_allclose(values, population**2)


def test_elitismo_mejor_historico_no_empeora():
    _, (_, _, _, best_history) = correr("x*sin(10*x)")
    mejores = [best_f for _, best_f in best_history]
    assert all(b <= a for a, b in zip(mejores, mejores[1:]))


def test_reproducible_con_semilla():
    _, r1 = correr("sin(x) + sin(10*x/3)", random_state=7)
    _, r2 = correr("sin(x) + sin(10*x/3)", random_state=7)
    assert r1[0] == r2[0] and r1[1] == r2[1]


def test_sigma_decay_y_tournament_k():
    ag, _ = correr("x**2", generations=10, sigma=0.1, sigma_decay=0.9, tournament_k=5)
    assert ag.tournament_k == 5
    assert ag.sigma == pytest.approx(0.1 * 20 * 0.9**10)


def test_dominio_invalido():
    with pytest.raises(ValueError):
        GeneticAlgorithm1D(build_function("x"), (5, 1))


def test_sin_individuos_validos_da_error_claro():
    with pytest.raises(ValueError, match="valores finitos"):
        correr("sqrt(x)", bounds=(-10, 0.001), pop_size=10, random_state=1)


# -----------------------------------------------------------
# Nichos (varios mínimos)
# -----------------------------------------------------------
def test_sin_nichos_no_hay_minimos_ni_cambia_el_resultado():
    ag, r1 = correr("(x**2 - 1)**2", bounds=(-2, 2))
    _, r2 = correr("(x**2 - 1)**2", bounds=(-2, 2), niching=False, niche_radius=0.2)
    assert ag.minimos() == [] and ag.niche_history == []
    assert r1[0] == r2[0] and r1[1] == r2[1]


def test_nichos_encuentran_los_dos_minimos_del_doble_pozo():
    ag, _ = correr("(x**2 - 1)**2", bounds=(-2, 2), niching=True, niche_radius=0.05)
    xs = sorted(m["x"] for m in ag.minimos() if m["f"] < 1e-4)
    assert len(xs) == 2
    assert xs[0] == pytest.approx(-1, abs=0.01) and xs[1] == pytest.approx(1, abs=0.01)


def test_nichos_encuentran_varios_minimos_globales_del_seno():
    # sin(2πx/3) en [-6, 6] tiene 4 mínimos globales (-1) en x = -3.75, -0.75, 2.25, 5.25
    ag, _ = correr("sin(2*pi*x/3)", bounds=(-6, 6), niching=True, niche_radius=0.05)
    globales = sorted(m["x"] for m in ag.minimos() if m["f"] < -0.99)
    esperados = [-3.75, -0.75, 2.25, 5.25]
    assert len(globales) == 4
    for x, esperado in zip(globales, esperados):
        assert x == pytest.approx(esperado, abs=0.05)


@pytest.mark.parametrize("expr,bounds", [
    ("x**2 + 10*cos(2*x) + x*sin(5*x)", (-10, 10)),
    ("0.3*sin(8*x) + 0.7*cos(15*x)", (-2, 2)),
])
def test_los_minimos_reportados_estan_en_el_fondo_de_un_valle(expr, bounds):
    ag, _ = correr(expr, bounds=bounds, niching=True, niche_radius=0.05)
    minimos = ag.minimos()
    assert minimos
    f = build_function(expr)
    paso = 0.05 * (bounds[1] - bounds[0]) / 2
    for m in minimos:
        for vecino in (m["x"] - paso, m["x"] + paso):
            vecino = min(max(vecino, bounds[0]), bounds[1])
            assert f(vecino) >= m["f"]
        assert m["individuos"] >= 1
    # Ordenados del mejor al peor y sin repetir nicho
    assert [m["f"] for m in minimos] == sorted(m["f"] for m in minimos)
    xs = sorted(m["x"] for m in minimos)
    assert all(b - a >= ag.niche_radius for a, b in zip(xs, xs[1:]))


def test_nichos_conservan_el_mejor_historico():
    ag, (best_x, best_f, _, best_history) = correr(
        "x**2 + 10*cos(2*x) + x*sin(5*x)", niching=True, niche_radius=0.05
    )
    mejores = [b for _, b in best_history]
    assert all(b <= a for a, b in zip(mejores, mejores[1:]))
    assert ag.minimos()[0]["f"] == pytest.approx(best_f, abs=1e-6)


def test_radio_de_nicho_invalido():
    with pytest.raises(ValueError, match="radio del nicho"):
        GeneticAlgorithm1D(build_function("x"), (0, 1), niching=True, niche_radius=0)


# -----------------------------------------------------------
# ^ como potencia y LaTeX
# -----------------------------------------------------------
@pytest.mark.parametrize("expr", ["9^9^9", "9^9^9 + x", "x + 9^(9^9)"])
def test_circunflejo_se_valida_como_potencia(expr):
    # En Python 9^9^9 se agrupa como (9^9)^9, pero sympify lo convierte en 9**(9**9)
    with pytest.raises(ValueError, match="demasiado grande"):
        build_function(expr)


@pytest.mark.parametrize(
    "expr, latex",
    [
        ("x^2 + 1", "x^{2} + 1"),
        ("(x**2 - 1)**2", r"\left(x^{2} - 1\right)^{2}"),
        ("1/(1 + exp(-10*x))", r"\frac{1}{1 + e^{-10 x}}"),
        ("log(abs(x) + 1)", r"\log\left(\left|x\right| + 1\right)"),
        ("sin(2*pi*x/3)", r"\sin\left(\frac{2 \pi x}{3}\right)"),
        ("x - (x + 1)", r"x - \left(x + 1\right)"),
        ("-x**2", "-x^{2}"),
        ("2*3", r"2 \cdot 3"),
    ],
)
def test_expresion_latex(expr, latex):
    assert expresion_latex(expr) == latex


def test_expresion_latex_valida_antes():
    with pytest.raises(ValueError):
        expresion_latex("__import__('os')")
