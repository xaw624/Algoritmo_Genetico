from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from AG_procesamiento import build_function

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at():
    """
    App lista para ejecutar rápido: pocas generaciones y animación mínima.
    """
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.number_input(key="generations").set_value(10)
    app.slider(key="frame_delay").set_value(10)
    return app


def ejecutar(app):
    app.sidebar.button[0].click().run()
    assert not app.exception
    return app


def metricas(app):
    return {m.label: m.value for m in app.metric}


def test_ejecucion_por_defecto(at):
    ejecutar(at)
    assert [s.value for s in at.success] == ["Simulación completada."]
    assert at.success[0].icon == "✅"
    assert metricas(at)["Generaciones evaluadas"] == "10"
    assert at.slider(key="gen_1").value == 10


def test_resultados_persisten_al_mover_un_control(at):
    ejecutar(at)
    antes = metricas(at)

    at.slider(key="pc").set_value(0.5).run()

    assert metricas(at) == antes
    assert not at.success  # no se repite la animación


def test_explorar_generacion(at):
    ejecutar(at)
    at.slider(key="gen_1").set_value(3).run()
    assert at.text[-1].value.startswith("Generación 3/10")


def test_limpiar_resultados(at):
    ejecutar(at)
    [b for b in at.button if b.label == "🗑️ Limpiar resultados"][0].click().run()
    assert not at.metric


def test_todos_los_ejemplos_son_validos(at):
    etiquetas = [b.label for b in at.main.button]
    assert len(etiquetas) == 12

    for etiqueta in etiquetas:
        [b for b in at.main.button if b.label == etiqueta][0].click().run()
        build_function(at.code[0].value)


def test_ejecucion_sin_semilla(at):
    at.checkbox(key="usar_semilla").uncheck()
    at.run()
    assert at.number_input(key="semilla").disabled

    ejecutar(at)
    assert "semilla: aleatoria" in at.caption[0].value


def test_dominio_invalido_borra_resultados(at):
    ejecutar(at)
    at.number_input(key="x_min").set_value(5.0)
    at.number_input(key="x_max").set_value(1.0)
    ejecutar(at)

    assert at.error[0].value == "El límite inferior debe ser menor que el límite superior."
    assert not at.metric


def test_expresion_maliciosa_no_se_ejecuta(at, tmp_path):
    marca = tmp_path / "ejecutado"
    at.text_input(key="func").set_value(f"__import__('pathlib').Path('{marca}').touch()")
    ejecutar(at)

    assert "Error al ejecutar la simulación" in at.error[0].value
    assert not marca.exists()
