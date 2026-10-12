import time

import numpy as np
import pytest

from AG_procesamiento import (
    MAX_LONGITUD_EXPRESION,
    GeneticAlgorithm1D,
    build_function,
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
