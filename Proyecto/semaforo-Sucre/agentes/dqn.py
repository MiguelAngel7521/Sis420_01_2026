
import copy
import random
import numpy as np
import torch


class DQN:
    def __init__(self, dim_estado, dim_accion, dim_oculta=64, lr=0.001):
        self.criterio = torch.nn.MSELoss()
        self.modelo = torch.nn.Sequential(
            torch.nn.Linear(dim_estado, dim_oculta),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_oculta, dim_oculta),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_oculta, dim_accion),
        )
        self.optimizador = torch.optim.Adam(self.modelo.parameters(), lr)

    def actualizar(self, estados, y): #respuestas correctas esperadas
        self.optimizador.zero_grad()
        prediccion = self.modelo(torch.from_numpy(np.array(estados, dtype=np.float32)))
        objetivo = torch.from_numpy(np.array(y, dtype=np.float32))
        perdida = self.criterio(prediccion, objetivo)
        perdida.backward()
        self.optimizador.step()
        return perdida.item()

    def predecir(self, estado):#as predicciones (los valores Q) para cada acción.
        with torch.no_grad():
            return self.modelo(torch.from_numpy(np.array(estado, dtype=np.float32)))

#Aprendia de una accion inmediata y despues la ol
class DQNReplay(DQN):
    def replay(self, memoria, tamano, gamma=0.95):
        if len(memoria) >= tamano:
            lote = random.sample(memoria, tamano)
            estados, acciones, sig_estados, recompensas, terminados = zip(*lote)
            estados = torch.from_numpy(np.array(estados, dtype=np.float32))
            acciones = torch.tensor(acciones, dtype=torch.long)
            sig_estados = torch.from_numpy(np.array(sig_estados, dtype=np.float32))
            recompensas = torch.from_numpy(np.array(recompensas, dtype=np.float32))
            terminados = torch.from_numpy(np.array(terminados, dtype=np.float32))

            q_vals = self.modelo(estados)
            q_sig = self.modelo(sig_estados)
            q_vals[range(len(q_vals)), acciones] = recompensas + gamma * (1 - terminados) * torch.max(q_sig, dim=1).values
            return self.actualizar(estados.numpy(), q_vals.detach().numpy())
        return None

#tiende a ser demasiado optimista,depende de el máximo valor estimado para calcular el futuro

class DQNDoble(DQN): #elige que accion es mejor el otro cuanto vale la accion
    def __init__(self, dim_estado, dim_accion, dim_oculta, lr):
        super().__init__(dim_estado, dim_accion, dim_oculta, lr)
        self.objetivo = copy.deepcopy(self.modelo)

    def actualizar_objetivo(self):
        self.objetivo.load_state_dict(self.modelo.state_dict())

    def replay(self, memoria, tamano, gamma=0.95):
        if len(memoria) >= tamano:
            lote = random.sample(memoria, tamano)
            estados, acciones, sig_estados, recompensas, terminados = zip(*lote)
            estados = torch.from_numpy(np.array(estados, dtype=np.float32))
            acciones = torch.tensor(acciones, dtype=torch.long)
            sig_estados = torch.from_numpy(np.array(sig_estados, dtype=np.float32))
            recompensas = torch.from_numpy(np.array(recompensas, dtype=np.float32))
            terminados = torch.from_numpy(np.array(terminados, dtype=np.float32))

            q_vals = self.modelo(estados)
            q_sig_online = self.modelo(sig_estados)
            mejores_acciones = torch.argmax(q_sig_online, dim=1) #escoge la mejor accion rojo o verde
            with torch.no_grad():
                q_sig_objetivo = self.objetivo(sig_estados)
            q_vals[range(len(q_vals)), acciones] = recompensas + gamma * (1 - terminados) * q_sig_objetivo[range(len(q_sig_objetivo)), mejores_acciones]
            return self.actualizar(estados.numpy(), q_vals.detach().numpy())
        return None


def cargar_modelo(modelo, ruta):
    modelo.modelo.load_state_dict(torch.load(ruta, map_location=torch.device("cpu")))
    modelo.modelo.eval()
    return modelo


def guardar_modelo(modelo, ruta):
    torch.save(modelo.modelo.state_dict(), ruta)
