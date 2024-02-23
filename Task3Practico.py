import numpy as np
import random
import matplotlib.pyplot as plt

# Constantes para el entorno
SIZE = 4
ACTIONS = ['up', 'down', 'left', 'right']
REWARD_GOAL = 1
REWARD_HOLE = -1

# Función para generar el entorno
def generate_environment(seed=None):
  """Genera un entorno Frozen Lake aleatorio.

  Args:
    seed: Semilla para la aleatoriedad.

  Returns:
    Un array 2D que representa el entorno.
  """
  if seed is not None:
    random.seed(seed)
  env = np.zeros((SIZE, SIZE), dtype=int)
  # Posición aleatoria del objetivo
  goal_x, goal_y = random.randint(0, SIZE-1), random.randint(0, SIZE-1)
  env[goal_x, goal_y] = REWARD_GOAL
  # Posiciones aleatorias de los hoyos (máximo 3)
  num_holes = random.randint(1, 3)
  for _ in range(num_holes):
    while True:
      hole_x, hole_y = random.randint(0, SIZE-1), random.randint(0, SIZE-1)
      if (hole_x, hole_y) != (goal_x, goal_y):
        break
    env[hole_x, hole_y] = REWARD_HOLE
  return env

# Función para obtener la recompensa de una acción
def get_reward(env, state, action):
  """Obtiene la recompensa de una acción en un estado dado.

  Args:
    env: El entorno.
    state: La posición actual del agente.
    action: La acción que se va a tomar.

  Returns:
    La recompensa de la acción.
  """
  next_state = get_next_state(state, action)
  return env[next_state[0], next_state[1]]

# Función para obtener el siguiente estado dado un estado y una acción
def get_next_state(state, action):
  """Obtiene el siguiente estado del agente dado un estado y una acción.

  Args:
    state: La posición actual del agente.
    action: La acción que se va a tomar.

  Returns:
    La siguiente posición del agente.
  """
  x, y = state
  if action == 'up':
    y = max(y - 1, 0)
  elif action == 'down':
    y = min(y + 1, SIZE-1)
  elif action == 'left':
    x = max(x - 1, 0)
  elif action == 'right':
    x = min(x + 1, SIZE-1)
  return (x, y)

# Función para calcular la política
def calculate_policy(env):
  """Calcula la política para el entorno dado.

  Args:
    env: El entorno.

  Returns:
    Un array 2D que representa la política.
  """
  policy = np.zeros((SIZE, SIZE), dtype=str)
  for x in range(SIZE):
    for y in range(SIZE):
      best_action = None
      best_reward = -np.inf
      for action in ACTIONS:
        reward = get_reward(env, (x, y), action)
        if reward > best_reward:
          best_reward = reward
          best_action = action
      policy[x, y] = best_action
  return policy

# Función para ejecutar la política
def run_policy(env, policy, start_state):
  """Ejecuta la política en el entorno desde un estado inicial.

  Args:
    env: El entorno.
    policy: La política.
    start_state: El estado inicial.

  Returns:
    La recompensa total y la secuencia de estados.
  """
  total_reward = 0
  states = [start_state]
  state = start_state
  while state != (3, 3):
    action = policy[state[0], state[1]]
    reward = get_reward(env, state, action)
    total_reward += reward
    next_state = get_next_state(state, action)
    state = next_state
    states.append(next_state)
    return total_reward, states

env = generate_environment()
policy = calculate_policy(env)

start_state = (0, 0)
total_reward, states = run_policy(env, policy, start_state)

print(f"Recompensa total desde {(0, 0)}: {total_reward}")
print(f"Estados: {states}")
