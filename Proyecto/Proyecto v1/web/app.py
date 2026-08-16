"""
WEB APP: VISUALIZACION DEL SEMAFORO INTELIGENTE
================================================

Muestra en el navegador la interseccion simulada con:
  - Animacion de autos llegando y pasando
  - Semáforo cambiando de color segun la fase
  - Graficas de colas en tiempo real
  - Comparacion contra semaforo fijo

Ejecutar con: python web/app.py
Abrir en: http://localhost:5000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request
from entorno import TrafficEnv
from agente import QLearningAgent
import numpy as np

app = Flask(__name__)

# Variables globales para la simulacion en vivo
env = TrafficEnv()
agente = QLearningAgent()
historial_simulacion = []
simulacion_activa = False
modo_actual = "inteligente"  # "inteligente" o "fijo"


def cargar_agente():
    """Intenta cargar un agente entrenado."""
    global agente
    try:
        ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'modelos', 'q_table.pickle')
        if os.path.exists(ruta):
            agente.cargar(ruta)
            return True
    except:
        pass
    return False


@app.route('/')
def index():
    """Pagina principal."""
    tiene_modelo = cargar_agente()
    return render_template('index.html',
                         tiene_modelo=tiene_modelo,
                         tamano_tabla=len(agente.Q) if tiene_modelo else 0)


@app.route('/api/iniciar', methods=['POST'])
def api_iniciar():
    """Inicia una nueva simulacion."""
    global env, historial_simulacion, simulacion_activa
    datos = request.get_json() or {}
    modo = datos.get('modo', 'inteligente')

    global modo_actual
    modo_actual = modo

    env = TrafficEnv()
    env.reset()
    historial_simulacion = []
    simulacion_activa = True

    return jsonify({'status': 'ok', 'mensaje': f'Simulación iniciada en modo {modo}'})


@app.route('/api/avanzar', methods=['POST'])
def api_avanzar():
    """Avanza la simulacion N pasos y devuelve el estado actual."""
    global env, historial_simulacion, simulacion_activa
    datos = request.get_json() or {}
    pasos = datos.get('pasos', 10)

    if not simulacion_activa:
        return jsonify({'status': 'error', 'mensaje': 'Simulación no iniciada'})

    cargar_agente()

    for _ in range(pasos):
        estado = env.obtener_estado()

        if modo_actual == "inteligente":
            accion = agente.elegir_accion(estado, explorar=False)
        else:
            # Semaforo fijo: cambia cada 30 segundos
            tiempo_fase = env.tiempo_fase
            accion = 1 if tiempo_fase >= 30 else 0

        nuevo_estado, recompensa, done, truncado, info = env.step(accion)

        historial_simulacion.append({
            'paso': env.pasos - 1,
            'colas': dict(env.colas),
            'fase': env.fase,
            'recompensa': recompensa,
            'accion': accion,
            'tiempo_fase': env.tiempo_fase
        })

        if truncado:
            simulacion_activa = False
            break

    # Devolver estado actual
    return jsonify({
        'status': 'ok',
        'paso': env.pasos,
        'colas': env.colas,
        'fase': env.fase,
        'tiempo_fase': env.tiempo_fase,
        'activa': simulacion_activa,
        'total_colas': sum(env.colas.values()),
        'historial': historial_simulacion[-pasos:]  # ultimos N pasos
    })


@app.route('/api/reiniciar', methods=['POST'])
def api_reiniciar():
    """Reinicia la simulacion."""
    global env, historial_simulacion, simulacion_activa
    env = TrafficEnv()
    env.reset()
    historial_simulacion = []
    simulacion_activa = False

    return jsonify({'status': 'ok'})


@app.route('/api/estado')
def api_estado():
    """Devuelve el estado actual de la simulacion."""
    global env, simulacion_activa
    return jsonify({
        'activa': simulacion_activa,
        'paso': env.pasos,
        'colas': env.colas,
        'fase': env.fase,
        'tiempo_fase': env.tiempo_fase,
        'total_colas': sum(env.colas.values()),
        'modo': modo_actual
    })


if __name__ == '__main__':
    print("=== SEMAFORO INTELIGENTE - WEB APP ===")
    print("Abriendo en http://localhost:5000")
    print()
    if cargar_agente():
        print(f"Agente cargado: {len(agente.Q)} estados aprendidos")
    else:
        print("ADVERTENCIA: No se encontro agente entrenado.")
        print("Ejecuta primero: python entrenar.py")
    app.run(debug=True)
