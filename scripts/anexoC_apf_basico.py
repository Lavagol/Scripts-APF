
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parámetros ----------------
k_att = 0.5        # constante atractiva
k_rep = 500           # constante repulsiva
dt = 1.0              # paso de integración
TOL = 1.0             # tolerancia de llegada
d0 = 5            # radio de influencia del obstáculo

obstacles = [np.array([40.0, 52.0]),
             #np.array([55.0, 47.0]),
             #np.array([65.0,55.0])
             ]

robot_pos = np.array([0.0, 50.0], dtype=float)
goal_pos  = np.array([80.0, 50.0], dtype=float)
trajectory: list[np.ndarray] = [robot_pos.copy()]
finished = False

repulsion_activada = [False] * len(obstacles)
entrada_d0         = [False] * len(obstacles)
posicion_evasion   = [None]  * len(obstacles)
distancia_evasion  = [None]  * len(obstacles)
posicion_minima    = [None]  * len(obstacles)
distancia_minima   = [np.inf]* len(obstacles)
posicion_inicio_d0 = [None]  * len(obstacles)

# ---------------- Gráfico ----------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set(xlim=(0, 100), ylim=(0, 100), title="Navegación APF", xlabel="X", ylabel="Y")
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

line,       = ax.plot([], [], 'b.-', label="Trayectoria")
robot_dot,  = ax.plot([], [], 'ko')
goal_dot,   = ax.plot(goal_pos[0], goal_pos[1], 'go', label="Objetivo")
start_dot,  = ax.plot(robot_pos[0], robot_pos[1], 'ro', label="Inicio")

# círculos de influencia y obstáculos fijos
distancias_marcadas = [2, 4, 6, 8, 10, 12, 14]
for obs in obstacles:
    ax.plot(*obs, 'ks', markersize=8)
    ax.add_patch(plt.Circle(obs, d0, edgecolor='black', facecolor='none', alpha=0.4))
    for r in distancias_marcadas:
        ax.add_patch(plt.Circle(obs, r, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.3))

# marcadores dinámicos: evasión (morado) y mínima distancia (naranja)
markers_evasion = [ax.plot([], [], 'mo',  markersize=8)[0] for _ in obstacles]
markers_min     = [ax.plot([], [], 'o',  color='orange', markersize=8)[0] for _ in obstacles]

# ---------------- Animación ----------------

def update(_frame):
    global robot_pos, finished
    if finished:
        return line, robot_dot, *markers_evasion, *markers_min

    # fuerza atractiva
    force_att = k_att * (goal_pos - robot_pos)

    # fuerza repulsiva total
    force_rep = np.zeros(2)
    for i, obs in enumerate(obstacles):
        direction = robot_pos - obs
        dist = np.linalg.norm(direction)

        if dist < distancia_minima[i]:
            distancia_minima[i] = dist
            posicion_minima[i] = robot_pos.copy()

        if dist <= d0 and not entrada_d0[i]:
            entrada_d0[i] = True
            posicion_inicio_d0[i] = robot_pos.copy()

        if dist <= d0 and not repulsion_activada[i]:
            repulsion_activada[i] = True
            distancia_evasion[i] = dist
            posicion_evasion[i] = robot_pos.copy()
            print(f"🌀 Evasión iniciada a {dist:.2f} m del obstáculo {i+1}")

        if dist <= d0:
            # evitar división por cero
            if dist == 0:
                dist = 1e-3
            magnitude = k_rep * ((1 / dist) - (1 / d0)) / dist**2
            force_rep += magnitude * (direction / dist)

    force_total = force_att + force_rep
    norm = np.linalg.norm(force_total)
    if norm > 1e-3:
        robot_pos[:] += (force_total / norm) * dt
        trajectory.append(robot_pos.copy())

    # llegada o colisión
    if np.linalg.norm(goal_pos - robot_pos) < TOL:
        print("✅ ¡Objetivo alcanzado!")
        finished = True
    for obs in obstacles:
        if np.linalg.norm(robot_pos - obs) < 2.0:
            print(f"❌ ¡COLISIÓN con {obs}!")
            finished = True

    # actualizar líneas
    traj = np.array(trajectory)
    line.set_data(traj[:, 0], traj[:, 1])
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])

    # actualizar marcadores de eventos
    for i in range(len(obstacles)):
        if posicion_evasion[i] is not None:
            markers_evasion[i].set_data([posicion_evasion[i][0]], [posicion_evasion[i][1]])
        if posicion_minima[i] is not None:
            markers_min[i].set_data([posicion_minima[i][0]], [posicion_minima[i][1]])
    return line, robot_dot, *markers_evasion, *markers_min

ani = FuncAnimation(fig, update, frames=500, interval=50, blit=True)
plt.show()

# ---------------- Resumen ----------------
traj = np.array(trajectory)
print("\n────────── RESULTADOS ──────────")
print(f"Pasos totales:           {len(traj):>3d}")
print(f"Longitud trayectoria:    {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.2f} m")
print(f"Error final (→ objetivo):{np.linalg.norm(goal_pos - traj[-1]):.2f} m")
for i in range(len(obstacles)):
    print((f"Obstáculo {i+1}:  inicio evasión a "
           f"{distancia_evasion[i] if distancia_evasion[i] is not None else 'inf'} m | "
           f"distancia mínima = {distancia_minima[i]:.2f} m"))
