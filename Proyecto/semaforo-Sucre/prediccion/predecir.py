"""Ejecuta un modelo DQN entrenado para controlar semaforos de Sucre."""

import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entorno.entorno_sucre import obtener_entorno
from agentes.dqn import DQNDoble, cargar_modelo
from configuracion.parametros_suma import (
    RUTA_MODELOS, EPISODIOS, OCULTOS, TASA_APRENDIZAJE
)


def linea_base_estatica(entorno):
    obs, info = entorno.reset()
    done = False
    truncado = False
    total = 0
    paso = 0
    while not (done or truncado):
        if paso % 4 == 0:
            accion = 1
        else:
            accion = 0
        obs, recompensa, done, truncado, info = entorno.step(accion)
        total += recompensa
        paso += 1
    print(f"Linea base estatica - Recompensa total: {total:.4f}")
    entorno.close()


def predecir_con_modelo(entorno, modelo):
    obs, info = entorno.reset()
    done = False
    truncado = False
    total = 0
    paso = 0
    while not (done or truncado):
        with torch.no_grad():
            q_vals = modelo.predecir(obs)
        accion = torch.argmax(q_vals).item()
        q0, q1 = q_vals[0].item(), q_vals[1].item()
        if paso % 12 == 0:
            print(f"t={entorno.sim_step:5.0f}s | Q(verde_H)=[{q0:+.4f}] Q(verde_V)=[{q1:+.4f}] -> accion={accion}")
        obs, recompensa, done, truncado, info = entorno.step(accion)
        total += recompensa
        paso += 1
    print(f"\nModelo DQN - Recompensa total: {total:.4f}  |  Pasos: {paso}")
    entorno.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predecir con modelo DQN para Sucre")
    parser.add_argument("--modo", choices=["linea_base", "modelo"], default="modelo",
                        help="Modo de ejecucion: linea_base (estatico) o modelo (DQN entrenado)")
    parser.add_argument("--modelo", default=os.path.join(RUTA_MODELOS, "dqn_sucre.pth"),
                        help="Ruta del modelo .pth")
    args = parser.parse_args()

    entorno = obtener_entorno(renderizar=True)

    if args.modo == "linea_base":
        linea_base_estatica(entorno)
    else:
        dim_estado = entorno.observation_space.shape[0]
        dim_accion = entorno.action_space.n
        modelo = DQNDoble(dim_estado, dim_accion, OCULTOS, TASA_APRENDIZAJE)
        modelo = cargar_modelo(modelo, args.modelo)
        predecir_con_modelo(entorno, modelo)


if __name__ == "__main__":
    main()
