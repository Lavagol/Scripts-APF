"""
APF básico (Anexo C)
--------------------
Simula un USV en un plano XY con un obstáculo fijo y calcula el movimiento por
Campos Potenciales Artificiales (APF). Muestra la animación y marca eventos:

- Entrada a d0 (radio de influencia del obstáculo).
- Punto de evasión (primer instante en que la fuerza repulsiva domina y se
  desvía de la línea recta al objetivo).
- Distancia mínima alcanzada respecto del obstáculo.

Parámetros principales:
- k_att: constante atractiva hacia la meta.
- k_rep: constante repulsiva del obstáculo.
- d0   : radio de influencia del obstáculo.
- dt   : paso temporal (s).
- TOL  : tolerancia de llegada (m).

Salidas visibles:
- Trayectoria animada, marcas de eventos y guardado opcional de métricas.

***Escenarios de las tablas, representado en Capítulo 5 tabla 5.3***
"""
import numpy as np  # Cálculo numérico con arrays
import matplotlib.pyplot as plt   # Gráficos  2D
from matplotlib.animation import FuncAnimation  # Animación cuadro a cuadro, en esta ocación se ocupó animación.

# ---------------- Parámetros ----------------
k_att = 0.5        # constante atractiva
k_rep = 500           # constante repulsiva
dt = 1.0              # paso de integración
TOL = 1.0             # tolerancia de llegada
d0 = 5            # radio de influencia del obstáculo


# Lista de obstáculos---------------------------------------------
obstacles = [np.array([40.0, 52.0]),
             #np.array([55.0, 47.0]),
             #np.array([65.0,55.0])
             ]
# -------------- Escenario (inicio, meta, obstáculos) --------------------
robot_pos = np.array([0.0, 50.0], dtype=float) # posición inicial del USV
goal_pos  = np.array([80.0, 50.0], dtype=float) # posición de la meta
trajectory: list[np.ndarray] = [robot_pos.copy()]# historial de posiciones para trazar
finished = False  # bandera para detener la animación cuando se llega o hay colisión, OJO SI HAY COLISIÓN LA SIMUÑACIÓN NO ENTREGA INFORMACIÓN Y SOLO SE QUEDE EN EJECUCIÓN

# Arreglos para registrar eventos por obstáculo (mismo largo que 'obstacles')
repulsion_activada = [False] * len(obstacles)  # si ya se activó repulsión por 1ª vez
entrada_d0         = [False] * len(obstacles)  # si ya se cruzó el círculo d0 por 1ª vez
posicion_evasion   = [None]  * len(obstacles)  # posición al inicio de la evasión
distancia_evasion  = [None]  * len(obstacles)  # distancia al obstáculo al iniciar la evasión
posicion_minima    = [None]  * len(obstacles)  # posición donde se logró la mínima distancia
distancia_minima   = [np.inf]* len(obstacles)  # valor de la mínima distancia alcanzada
posicion_inicio_d0 = [None]  * len(obstacles)  # posición exacta al entrar en d0 por 1ª vez

# ---------------- Gráfico ----------------
fig, ax = plt.subplots(figsize=(8, 8))                  # crea figura y ejes
ax.set(xlim=(0, 100), ylim=(0, 100), title="Navegación APF", xlabel="X", ylabel="Y") # límites y etiquetas
ax.set_aspect('equal', adjustable='box')  # escalas iguales en X e Y
ax.grid(True)                               # grilla de referencia


# Elementos gráficos que se actualizarán:
line,       = ax.plot([], [], 'b.-', label="Trayectoria")                   # línea azul de trayectoria
robot_dot,  = ax.plot([], [], 'ko')                                         # punto negro = USV
goal_dot,   = ax.plot(goal_pos[0], goal_pos[1], 'go', label="Objetivo")     # punto verde = meta
start_dot,  = ax.plot(robot_pos[0], robot_pos[1], 'ro', label="Inicio")     # punto rojo = inicio

# círculos de influencia y obstáculos fijos
distancias_marcadas = [2, 4, 6, 8, 10, 12, 14]  #radios extra para visualización EN METROS
for obs in obstacles:                
    ax.plot(*obs, 'ks', markersize=8)  # dibuja el obstáculo como cuadrado negro
    ax.add_patch(plt.Circle(obs, d0, edgecolor='black', facecolor='none', alpha=0.4))  # círculo d0
    for r in distancias_marcadas:
        ax.add_patch(plt.Circle(obs, r, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.3))   # círculos grises punteados (ayuda a leer distancias en la animación)

# marcadores dinámicos: evasión (morado) y mínima distancia (naranja), EL IMPORTANTE ES EL NARANJO, sin embargo,  EL INCIO DE EVASIÓN SOLO CORROBORA LA FÓRMULA DE REPULSIÓN
markers_evasion = [ax.plot([], [], 'mo',  markersize=8)[0] for _ in obstacles]
markers_min     = [ax.plot([], [], 'o',  color='orange', markersize=8)[0] for _ in obstacles]

# ---------------- Animación ----------------

def update(_frame):

    """
    Un paso de simulación/animación:
      1) Calcula fuerzas atractiva y repulsiva.
      2) Actualiza la posición del USV (integración con dt).
      3) Detecta/guarda eventos (entrada a d0, evasión, mínima distancia).
      4) Actualiza los elementos gráficos (trayectoria, marcadores).
    """
    global robot_pos, finished   # se modifica el estado global del robot y bandera de fin
    if finished:
        return line, robot_dot, *markers_evasion, *markers_min

    # fuerza atractiva
    force_att = k_att * (goal_pos - robot_pos)

    # fuerza repulsiva total
    force_rep = np.zeros(2)
    for i, obs in enumerate(obstacles):
        direction = robot_pos - obs       # vector de alejamiento (desde obstáculo al robot)
        dist = np.linalg.norm(direction)  # distancia actual al obstáculo i

        # Actualización de distancia mínima y su posición asociada
        if dist < distancia_minima[i]:
            distancia_minima[i] = dist
            posicion_minima[i] = robot_pos.copy()

        # Registrar la primera entrada al círculo d0 (evento)
        if dist <= d0 and not entrada_d0[i]:
            entrada_d0[i] = True
            posicion_inicio_d0[i] = robot_pos.copy()

        # Registrar el inicio de la evasión (1ª vez que hay repulsión activa)
        if dist <= d0 and not repulsion_activada[i]:
            repulsion_activada[i] = True
            distancia_evasion[i] = dist
            posicion_evasion[i] = robot_pos.copy()
            print(f"🌀 Evasión iniciada a {dist:.2f} m del obstáculo {i+1}")

        # Si está dentro de d0, aplicar la repulsión del modelo por literatura
        if dist <= d0:
            # evitar división por cero, si dist==0
            if dist == 0:
                dist = 1e-3
            magnitude = k_rep * ((1 / dist) - (1 / d0)) / dist**2    # magnitud de la repulsión
            force_rep += magnitude * (direction / dist)              # dirección repulsiva = unit(robot_pos - obs); se suma al total

    # Fuerza total = atractiva + repulsiva, trabajaremos fuerza en ingles
    force_total = force_att + force_rep
    norm = np.linalg.norm(force_total) #normalización de la dirección
    
    # Si hay fuerza neta, avanzar en su dirección una distancia respecto al dt,
    if norm > 1e-3:
        robot_pos[:] += (force_total / norm) * dt  # actualización in-place de la posición
        trajectory.append(robot_pos.copy())        # guardar en historial para graficar

    # llegada o colisión
    if np.linalg.norm(goal_pos - robot_pos) < TOL:
        print("✅ ¡Objetivo alcanzado!")
        finished = True

     # Colisión: si el robot se acerca por debajo de un umbral establecido en este caso 2 metros   
    for obs in obstacles:
        if np.linalg.norm(robot_pos - obs) < 2.0:
            print(f"❌ ¡COLISIÓN con {obs}!")
            finished = True

    # --- Actualización de la curva de trayectoria y del punto del USV ---
    traj = np.array(trajectory)  # convierte lista de puntos a array 
    line.set_data(traj[:, 0], traj[:, 1])  # curva azul de la trayectoria
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])  # punto negro en la posición actual

    # actualizar marcadores de eventos
    for i in range(len(obstacles)):
        if posicion_evasion[i] is not None:
            markers_evasion[i].set_data([posicion_evasion[i][0]], [posicion_evasion[i][1]])
        if posicion_minima[i] is not None:
            markers_min[i].set_data([posicion_minima[i][0]], [posicion_minima[i][1]])
    return line, robot_dot, *markers_evasion, *markers_min

# Crea la animación: 500 frames, intervalo 50 ms, blitting( técnica para copiar bloques de pixeles)
ani = FuncAnimation(fig, update, frames=500, interval=50, blit=True)
plt.show()

# ---------------- Resumen del print al cerrar la simulación del malplotlib animado ----------------
traj = np.array(trajectory)   # trayectoria completa como array
print("\n────────── RESULTADOS ──────────")
print(f"Pasos totales:           {len(traj):>3d}") # número de posiciones guardadas, osea cada iteración
# longitud total de la curva: suma de normas de diferencias consecutivas
print(f"Longitud trayectoria:    {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.2f} m")
# error final a la meta (distancia directa desde el último punto), mmm se dejó como error, porque es lo que falta al objetivo
print(f"Error final (→ objetivo):{np.linalg.norm(goal_pos - traj[-1]):.2f} m")

# Para cada obstáculo, reporta distancia de inicio de evasión (do) y mínima alcanzada (PMA)
for i in range(len(obstacles)):
    print((f"Obstáculo {i+1}:  inicio evasión a "
           f"{distancia_evasion[i] if distancia_evasion[i] is not None else 'inf'} m | "
           f"distancia mínima = {distancia_minima[i]:.2f} m"))
