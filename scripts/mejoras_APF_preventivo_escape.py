
"""
APF → Preventivo → Escape lateral de emergencia con hold
Mide el “avance” (distancia recorrida) en cada modo:
  • APF puro
  • Push preventivo
  • Escape lateral de emergencia (mantiene 90° durante D_lat m)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parámetros ----------------
k_att, k_rep     = 1.0, 3500.0    # ganancia atractiva y repulsiva
dt, TOL          = 1.0, 1.0       # paso temporal y tolerancia llegada
d0               = 10.0           # radio de influencia
v_max            = 2.5            # velocidad máxima

#parámetro fuerza tg
k_tan            = 300.0          # cte tangente
d_pre            = 15.0           # radio pre‑emptivo
angle_pre_deg    = 8.0            # ángulo frontal pre‑emptivo



cos_pre          = np.cos(np.deg2rad(angle_pre_deg))

# Escape lateral
D_lat            = 20.0                            # distancia total de hold lateral (m)
lat_steps        = int(D_lat / (v_max * dt))       # nº de frames que dura el hold
safety_dist      = 4.0                             # umbral de colisión para predecir

# Inercia (momentum)
v_prev           = np.zeros(2)
alpha            = 0.30           # fracción de inercia (0=no inercia, 1=solo inercia)

# ---------------- Escenario ----------------
obstacles = [
    
    
    np.array([53.0, 50.0]),
    #np.array([40.0, 50.0]),
    #np.array([40.0, 45.0])
]
robot_pos   = np.array([20.0, 80.0])
goal_pos    = np.array([80.0, 20.0])
trajectory  = [robot_pos.copy()]
finished    = False

# Variables de evento y contadores
rep_on            = [False]*len(obstacles)
dist_min          = [np.inf]*len(obstacles)
avance_apf        = 0.0
avance_preemptivo = 0.0
avance_lat        = 0.0

# Estado lateral
lat_counter = 0   #Fin del hold, el algoritmo vuelve al flujo normal
lat_dir     = np.zeros(2)

# ---------------- Gráfico ----------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set(xlim=(0,100), ylim=(0,100),
       title="Navegación APF",
       xlabel="X", ylabel="Y")
ax.set_aspect('equal',adjustable='box'); 
ax.grid(True)

# Objetivo y obstáculos
distancias_marcadas = [2, 4, 6, 8, 10, 12, 14]
ax.plot(*goal_pos, 'go', label="Objetivo")
for obs in obstacles:
    ax.plot(*obs, 'ks', markersize=8)
    ax.add_patch(plt.Circle(obs, d0, edgecolor='black', facecolor='none', alpha=0.4))
    for r in distancias_marcadas:
        ax.add_patch(plt.Circle(obs, r, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.3))

# marcadores dinámicos: evasión (morado) y mínima distancia (naranja)
markers_evasion = [ax.plot([], [], 'mo',  markersize=8)[0] for _ in obstacles]
markers_min     = [ax.plot([], [], 'o',  color='orange', markersize=8)[0] for _ in obstacles]

# Trayectoria y marcador de escape lateral
line,      = ax.plot([], [], 'b.-', label="Trayectoria")
robot_dot, = ax.plot([], [], 'ko',  label="Robot")
marker_lat = ax.plot([], [], '*', c='cyan', ms=12, label="Escape lateral")[0]
goal_dot,   = ax.plot(goal_pos[0], goal_pos[1], 'go')  # sin label duplicado
start_dot,  = ax.plot(robot_pos[0], robot_pos[1], 'ro', label="Inicio")
def update(_):
    global robot_pos, finished, v_prev, lat_counter, lat_dir
    global avance_apf, avance_preemptivo, avance_lat

    # 1) Si ya terminamos
    if finished:
        marker_lat.set_data([], [])
        return (line, robot_dot, marker_lat)

    # 2) Hold lateral si estamos en modo de emergencia
    if lat_counter > 0:
        robot_pos[:] += lat_dir * v_max * dt
        avance_lat  += v_max * dt
        lat_counter -= 1
        marker_lat.set_data([robot_pos[0]], [robot_pos[1]])
    else:
        marker_lat.set_data([], [])

        # 3) Cálculo fuerzas APF puro
        F_att = k_att * (goal_pos - robot_pos)
        F_rep = np.zeros(2)
        dist_list = []
        for obs in obstacles:
            vec = robot_pos - obs
            d   = np.linalg.norm(vec)
            dist_list.append(d)
            if d <= d0 and d > 1e-6:
                F_rep += k_rep * ((1/d) - (1/d0)) / d**2 * (vec/d)

        F_tot    = F_att + F_rep
        att_mag  = np.linalg.norm(F_att)
        goal_dir = (goal_pos - robot_pos)
        goal_dir /= np.linalg.norm(goal_dir)

        # 4) Push pre‑emptivo
        mode  = 'apf'
        F_tan = np.zeros(2)
        if att_mag > 1e-6:
            hdir = F_att / att_mag
            for obs, d in zip(obstacles, dist_list):
                if 1e-6 < d < d_pre:
                    u_r    = (robot_pos - obs) / d
                    cos_th = np.dot(hdir, -u_r)
                    γ      = np.clip((cos_th - cos_pre) / (1 - cos_pre), 0, 1)
                    β      = (d_pre - d) / d_pre
                    k_eff  = k_tan * β * γ
                    if k_eff > 0:
                        u_t1 = np.array([-u_r[1], u_r[0]])
                        u_t2 = np.array([ u_r[1], -u_r[0]])
                        u_t  = u_t1 if np.dot(u_t1, goal_dir) > np.dot(u_t2, goal_dir) else u_t2
                        F_tan += k_eff * u_t
                        mode   = 'pre'
                        break

        # 5) Predicción de colisión → Initiate lateral escape si es necesario
        F_move = F_tot + F_tan
        normM  = np.linalg.norm(F_move)
        if normM > 1e-6:
            direction = F_move / normM
            speed     = min(v_max, normM)
        else:
            direction, speed = np.zeros(2), 0.0

        
        next_pos = robot_pos + direction * speed * dt
        
        #Se dispara solo si estoy en modo APF(no preventivo), con la poscion para el siguiente paso next_pos caerís menos de safety_dist de algún obtáculo
        if mode == 'apf' and any(np.linalg.norm(next_pos-obs) < safety_dist for obs in obstacles):
            #Calcula dos direcciones laterales ortogonales al rumbo al objetivo (goal_dir) +-90°
            u_up = np.array([-goal_dir[1], goal_dir[0]])
            u_dn = -u_up
            #Evalúa cuál lado aleja más de los obstáculos con un “score”:
            def score(u): #Elige la dirección con menor score (equivalente a mayor distancia agregada).
                p = robot_pos + u * v_max * dt
                return sum(1/np.linalg.norm(p-obs) for obs in obstacles)
            
            #Inicializa el hold lateral
            lat_dir     = u_up if score(u_up) < score(u_dn) else u_dn
            lat_counter = lat_steps
            # Da el primer paso lateral inmediatamente (no espera al siguiente frame)
            robot_pos[:] += lat_dir * v_max * dt
            avance_lat  += v_max * dt
            marker_lat.set_data([robot_pos[0]], [robot_pos[1]])
        else:
            # 6) Aquí reinsertamos el bloque de inercia antes de movernos
            if np.linalg.norm(v_prev) > 1e-6:
                dir_prev  = v_prev / np.linalg.norm(v_prev)
                direction = (1 - alpha) * direction + alpha * dir_prev
                direction /= np.linalg.norm(direction)
            v_prev = direction * speed

            # 7) Movimiento final (APF o preventivo)
            robot_pos[:] += direction * speed * dt
            if mode == 'apf':
                avance_apf        += speed * dt
            else:
                avance_preemptivo += speed * dt

    trajectory.append(robot_pos.copy())

    # 8) Condición de fin y repintado
    if (np.linalg.norm(goal_pos - robot_pos) < TOL or
        any(np.linalg.norm(robot_pos-obs) < 2 for obs in obstacles)):
        finished = True

    traj = np.array(trajectory)
    line.set_data(traj[:,0], traj[:,1])
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    return (line, robot_dot, marker_lat)

# Arrancar animación
ani = FuncAnimation(fig, update, frames=400, interval=50, blit=True)
plt.legend(); plt.show()

# Resumen final
traj = np.array(trajectory)
print("\nRESULTADOS:")
print(f"Pasos: {len(traj)}  Distancia: {np.sum(np.linalg.norm(np.diff(traj,axis=0),axis=1)):.2f} m")
print(f"APF:        {avance_apf:.2f} m")
print(f"Preventivo:     {avance_preemptivo:.2f} m")
print(f"Escape lateral:  {avance_lat:.2f} m")
