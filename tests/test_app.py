from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from AG_procesamiento import build_function

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at():
    """
    App lista para ejecutar rápido: pocas generaciones y animación rápida.
    """
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.number_input(key="generations").set_value(10)
    app.radio(key="velocidad").set_value("Rápida")
    app.run()
    return app


def ejecutar(app):
    app.button(key="ejecutar").click().run()
    assert not app.exception
    return app


def metricas(app):
    return {m.label: m.value for m in app.metric}


def mensajes(app, tipo):
    return [m.value for m in getattr(app, tipo)]


def formula(app):
    # AppTest devuelve el LaTeX envuelto en $$ ... $$
    return app.latex[0].value.strip("$\n")


# -----------------------------------------------------------
# Estado inicial
# -----------------------------------------------------------
def test_pantalla_inicial(at):
    assert not at.exception
    assert at.button(key="ejecutar").disabled is False
    assert formula(at).startswith("f(x) = x^{2}")
    assert len(at.get("image")) == 1  # vista previa de la curva
    assert not at.metric


def test_ejecutar_esta_en_el_area_principal(at):
    assert at.button(key="ejecutar") in list(at.main.button)
    assert not list(at.sidebar.button)


# -----------------------------------------------------------
# Ejecución y resultados
# -----------------------------------------------------------
def test_ejecucion_por_defecto(at):
    ejecutar(at)
    assert mensajes(at, "success") == ["Simulación completada."]
    assert at.success[0].icon == "✅"
    assert metricas(at)["Generaciones evaluadas"] == "10"
    assert at.slider(key="gen_1").value == 10


def test_ejecucion_sin_animacion(at):
    at.checkbox(key="animar").uncheck().run()
    assert at.radio(key="velocidad").disabled

    ejecutar(at)
    assert metricas(at)["Generaciones evaluadas"] == "10"
    assert at.text[-1].value.startswith("Generación 10/10")


def test_resultados_persisten_y_avisan_si_cambia_un_parametro(at):
    ejecutar(at)
    antes = metricas(at)
    assert not at.info

    at.slider(key="pc").set_value(0.5).run()

    assert metricas(at) == antes
    assert not at.success  # no se repite la animación
    assert "Cambiaste parámetros" in at.info[0].value


def test_cambiar_el_problema_muestra_vista_previa(at):
    ejecutar(at)
    at.selectbox(key="ejemplo").set_value("Parábola").run()

    assert not at.metric
    assert "La función o el dominio cambiaron" in at.info[0].value

    # Al volver al problema original reaparecen los resultados
    at.selectbox(key="ejemplo").set_value("Combinada").run()
    assert metricas(at)["Generaciones evaluadas"] == "10"


def test_explorar_generacion(at):
    ejecutar(at)
    at.slider(key="gen_1").set_value(3).run()
    assert at.text[-1].value.startswith("Generación 3/10")


def test_limpiar_resultados(at):
    ejecutar(at)
    [b for b in at.button if b.label == "🗑️ Limpiar resultados"][0].click().run()
    assert not at.metric


def test_ejecucion_sin_semilla(at):
    at.checkbox(key="usar_semilla").uncheck()
    at.run()
    assert at.number_input(key="semilla").disabled

    ejecutar(at)
    assert "semilla: aleatoria" in at.caption[-1].value or any(
        "semilla: aleatoria" in c.value for c in at.caption
    )


# -----------------------------------------------------------
# Ejemplos
# -----------------------------------------------------------
def test_todos_los_ejemplos_son_validos_y_cargan_su_dominio(at):
    opciones = at.selectbox(key="ejemplo").options
    ejemplos = [o for o in opciones if not o.startswith("—")]
    assert len(ejemplos) == 13

    for nombre in ejemplos:
        at.selectbox(key="ejemplo").set_value(nombre).run()

        expr = at.text_input(key="func").value
        build_function(expr)
        assert at.number_input(key="x_min").value < at.number_input(key="x_max").value
        assert not at.error, nombre
        assert at.button(key="ejecutar").disabled is False
        assert at.caption[0].value  # descripción del ejemplo


def test_doble_pozo_carga_dominio_sugerido(at):
    at.selectbox(key="ejemplo").set_value("Doble pozo").run()
    assert at.text_input(key="func").value == "(x**2 - 1)**2"
    assert at.number_input(key="x_min").value == -2.0
    assert at.number_input(key="x_max").value == 2.0


def test_editar_la_funcion_pasa_a_personalizada(at):
    at.text_input(key="func").set_value("x**4 - x").run()
    assert at.selectbox(key="ejemplo").value.startswith("—")
    assert formula(at) == "f(x) = x^{4} - x"


# -----------------------------------------------------------
# Validación en vivo
# -----------------------------------------------------------
def test_dominio_invalido_se_avisa_sin_ejecutar(at):
    ejecutar(at)
    at.number_input(key="x_min").set_value(5.0)
    at.number_input(key="x_max").set_value(1.0)
    at.run()

    assert mensajes(at, "error") == ["El límite inferior debe ser menor que el límite superior."]
    assert at.button(key="ejecutar").disabled
    assert not at.metric


def test_funcion_invalida_se_avisa_sin_ejecutar(at):
    at.text_input(key="func").set_value("y + x").run()
    assert "Nombre no permitido" in at.error[0].value
    assert at.button(key="ejecutar").disabled


def test_funcion_sin_valores_en_el_dominio(at):
    at.text_input(key="func").set_value("log(x)").run()
    at.number_input(key="x_min").set_value(-10.0)
    at.number_input(key="x_max").set_value(-1.0)
    at.run()
    assert "no tiene valores válidos" in at.error[0].value


def test_expresion_maliciosa_no_se_ejecuta(at, tmp_path):
    marca = tmp_path / "ejecutado"
    at.text_input(key="func").set_value(f"__import__('pathlib').Path('{marca}').touch()").run()

    assert at.error
    assert at.button(key="ejecutar").disabled
    assert not marca.exists()
