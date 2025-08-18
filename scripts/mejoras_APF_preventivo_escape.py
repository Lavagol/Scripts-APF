"""
APF con mejoras: preventivo + escape lateral
--------------------------------------------
Extiende el APF básico con dos comportamientos para mitigar mínimos locales y
mejorar la seguridad:

1) Preventivo (fuerza tangencial)
   - Activa cuando un obstáculo está en sector frontal y a distancia < d_pre.
   - Se suma una componente tangencial ±90° respecto al vector de alejamiento del obstáculo
     (se elige el lado con mejor trayectoria hacia la meta). Esto “rodea” el obstáculo y reduce
     el riesgo de quedar atrapado cerca de d0.

2) Escape lateral con hold (emergencia)
   - Si la posición prevista del siguiente paso cae por debajo de una distancia de seguridad
     (safety_dist) respecto a algún obstáculo y el modo actual es APF, se fija un rumbo
     lateral ±90° durante una distancia objetivo D_lat (hold) y luego se vuelve al flujo normal.

Velocidad (speed) y límite v_max
--------------------------------
En APF/Preventivo: se calcula F_move = F_tot + F_tan y
    speed = min(||F_move||, v_max)
Por lo tanto, el avance por frame es Δs = speed * dt ≤ v_max * dt.
En Escape lateral: se mueve con
    speed = v_max  y  rumbo = ±90°,
y la duración del hold está dada por:
    lat_steps = int(D_lat / (v_max * dt))

(Nota: No se usa filtro de inercia/alpha en la versión final.)
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parámetros, juntamos con (,) en esta ocación, para separar los parámetros por tipo de avance----------------
k_att, k_rep     = 1.0, 3500.0    # cte atractiva y repulsiva
dt, TOL          = 1.0, 1.0       # paso temporal y tolerancia llegada
d0               = 10.0           # radio de influencia


#parámetro fuerza tg (modo preventivo), los valores en esta ocación se varían ya que son para probar la evasión preventiva al obstáculo y analizar los comportamientos
k_tan            = 300.0          # cte tangente
d_pre            = 15.0           # radio pre‑emptivo
angle_pre_deg    = 8.0            # ángulo frontal pre‑emptivo
#función que calcula el coseno y np.deg2rad convierte angulo en radianes. ( es el umbral en coseno)
cos_pre          = np.cos(np.deg2rad(angle_pre_deg)) # umbral en coseno para el criterio angular

# Escape lateral hold= escape de emergencia, tiene nombre hold porque actúa sin restricciones, dependiendo de la distancia que se establece
v_max            = 2.5            # velocidad máxima, para el escape de emergencia 
D_lat            = 20.0                            # distancia total de hold lateral (m)
lat_steps        = int(D_lat / (v_max * dt))       # nº de frames que dura el hold o escape de emergencia
safety_dist      = 4.0                             # umbral de colisión para predecir

# Inercia (momentum), no se ocupa!!!!!!!! 
v_prev           = np.zeros(2)
alpha            = 0.30           # fracción de inercia (0=no inercia, 1=solo inercia)

# ---------------- Escenario ----------------
obstacles = [

    np.array([53.0, 50.0]),
    #np.array([40.0, 50.0]),
    #np.array([40.0, 45.0])
]
robot_pos   = np.array([20.0, 80.0])  # posición inicial del robot
goal_pos    = np.array([80.0, 20.0])  # posición de la meta
trajectory  = [robot_pos.copy()]      # historial de posiciones (para trazar trayectoria)
finished    = False                   # bandera de término (llegada/colisión)

# Variables de evento y contadores (algunas no usadas en esta versión, pero mantenidas)
rep_on            = [False]*len(obstacles)   # (no usada) flag por obstáculo
dist_min          = [np.inf]*len(obstacles)  # (no usada) mínima distancia por obstáculo
# Métricas de distancia recorrida por modo (para el reporte final)
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
ax.set_aspect('equal',adjustable='box');   # misma escala en X e Y
ax.grid(True)

# Objetivo y obstáculos, MISMO QUE EL ANEXO C.
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

# ---------------- Bucle de animación / actualización ----------------
def update(_):
    """
    Un paso de simulación:
      - Si está en hold lateral, avanza lateralmente (±90° respecto a la línea a la meta).
      - Si no, calcula fuerzas APF, añade preventivo si corresponde, y predice colisión.
      - Si hay riesgo y está en APF, activa escape lateral (y da el primer paso).
      - Aplica inercia (si corresponde) y mueve el robot, en esta ocación y para trabajo tesis no se considerará
      - Actualiza trayectoria, condiciones de término y gráficos.
    """
    global robot_pos, finished, v_prev, lat_counter, lat_dir
    global avance_apf, avance_preemptivo, avance_lat

    # 1) Si ya terminamos, solo mantener para el blitting (tipo de transferencia de bloques de bits).
    if finished:
        marker_lat.set_data([], [])
        return (line, robot_dot, marker_lat)

    # 2) Hold lateral si estamos en modo de emergencia (rumbo fijo ±90°), lo ideal es no activarlo
    if lat_counter > 0:
        robot_pos[:] += lat_dir * v_max * dt     # avanzar en dirección lateral fija
        avance_lat  += v_max * dt                # acumular distancia en modo lateral
        lat_counter -= 1                         # descontar un frame del hold
        marker_lat.set_data([robot_pos[0]], [robot_pos[1]])  # marcar en la figura
    else:
        # Si no estamos en hold lateral, ocultar marcador
        marker_lat.set_data([], [])

        # 3) Cálculo fuerzas APF , sa cambio force por F_...
        F_att = k_att * (goal_pos - robot_pos)   # atractiva hacia la meta
        F_rep = np.zeros(2)                      # acumulador repulsivo
        dist_list = []                           # distancias a cada obstáculo (para reuso)

        for obs in obstacles:
            vec = robot_pos - obs                # vector de alejamiento (robot - obstáculo)
            d   = np.linalg.norm(vec)            # distancia actual al obstáculo
            dist_list.append(d)
            # repulsión activa si d ≤ d0
            if d <= d0 and d > 1e-6:
                F_rep += k_rep * ((1/d) - (1/d0)) / d**2 * (vec/d)

        F_tot    = F_att + F_rep            # fuerza total APF
        att_mag  = np.linalg.norm(F_att)    # normalización de la atractiva
        goal_dir = (goal_pos - robot_pos)   # dirección hacia la meta (unitaria)
        goal_dir /= np.linalg.norm(goal_dir)

        # 4) PREVENTIVO (fuerza tangencial): activa si obstáculo frontal/lateral y d < d_pre
        mode  = 'apf'                            # modo actual por defecto
        F_tan = np.zeros(2)                      # acumulador tangencial
        if att_mag > 1e-6:
            hdir = F_att / att_mag               # “rumbo deseado” según atractiva pura
            for obs, d in zip(obstacles, dist_list):
                if 1e-6 < d < d_pre:
                    u_r    = (robot_pos - obs) / d    # unitario de alejamiento (radial)
                    # similitud entre hdir y (-u_r): grande → obstáculo está en frente
                    cos_th = np.dot(hdir, -u_r)
                     # γ (gamma) pondera por alineación frontal (0..1). Si cos_th ≤ cos_pre entonces es 0; si ≈1 entonces es 1
                    γ      = np.clip((cos_th - cos_pre) / (1 - cos_pre), 0, 1)
                    β      = (d_pre - d) / d_pre
                    # k_eff: magnitud efectiva del empuje o fuerza tangencial
                    k_eff  = k_tan * β * γ
                    if k_eff > 0:
                        # Genera las dos ortogonales a u_r: izquierda  y derecha 
                        u_t1 = np.array([-u_r[1], u_r[0]])    # +90° (izquierda)
                        u_t2 = np.array([ u_r[1], -u_r[0]])   # -90° (derecha)
                        # Elegir la  mejor hacia la meta (mayor proyección sobre goal_dir)
                        u_t  = u_t1 if np.dot(u_t1, goal_dir) > np.dot(u_t2, goal_dir) else u_t2
                        F_tan += k_eff * u_t                  # sumar tangencial
                        mode   = 'pre'                        # estamos en modo preventivo
                        break  # con un obstáculo “dominante” basta para activar preventivo

        # 5) Predicción de colisión: si en APF puro next_pos cae < safety_dist → activar escape lateral 
        F_move = F_tot + F_tan     # fuerza final a convertir en movimiento
        normM  = np.linalg.norm(F_move)
        if normM > 1e-6:
            direction = F_move / normM           # dirección unitaria de movimiento
            speed     = min(v_max, normM)        # velocidad limitada por v_max, El tema de la valocidad debo seguir pensando bien , que establecer pero como es emergencia no le tomé tanta importancia
        else:
            direction, speed = np.zeros(2), 0.0 # sin fuerza → no se mueve

        
        next_pos = robot_pos + direction * speed * dt  # posición prevista para el próximo paso
        
        #Se dispara solo si estoy en modo APF(no preventivo), con la poscion para el siguiente paso next_pos caerís menos de safety_dist de algún obtáculo
        if mode == 'apf' and any(np.linalg.norm(next_pos-obs) < safety_dist for obs in obstacles):
            #Calcula dos direcciones laterales ortogonales al rumbo al objetivo (goal_dir) +-90°
            u_up = np.array([-goal_dir[1], goal_dir[0]])
            u_dn = -u_up
            #Evalúa cuál lado aleja más de los obstáculos con un “score”(menor suma (1/dist))
            def score(u): #Elige la dirección con menor score (equivalente a mayor distancia agregada).
                p = robot_pos + u * v_max * dt   # punto si doy un paso lateral
                return sum(1/np.linalg.norm(p-obs) for obs in obstacles)
            
            #Inicializa el hold o escape lateral de emergencia
            lat_dir     = u_up if score(u_up) < score(u_dn) else u_dn
            lat_counter = lat_steps
            # Da el primer paso lateral inmediatamente (no espera al siguiente frame)
            robot_pos[:] += lat_dir * v_max * dt
            avance_lat  += v_max * dt
            marker_lat.set_data([robot_pos[0]], [robot_pos[1]])
        else:
            #  6) INERCIA: mezcla con la dirección previa para suavizar cambios bruscos,  pero para trabajo final no consideraremos inercia
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
    # Guardar la nueva posición en la trayectoria (para dibujar/medir)
    trajectory.append(robot_pos.copy())

    # 8) Condición de fin y repintado llegada a la meta o colisión (distancia < 2 m a cualquier obstáculo), 2 metros establecido solo por pruebas y ver comportamiento
    if (np.linalg.norm(goal_pos - robot_pos) < TOL or
        any(np.linalg.norm(robot_pos-obs) < 2 for obs in obstacles)):
        finished = True
    # Actualizar objetos gráficos (trayectoria y posición actual)
    traj = np.array(trajectory)
    line.set_data(traj[:,0], traj[:,1])
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    return (line, robot_dot, marker_lat)

# Arrancar animación: 400 frames, cada 50 ms, lo mismo que el trabajo anterior
ani = FuncAnimation(fig, update, frames=400, interval=50, blit=True)
plt.legend(); plt.show()

# Resumen final al cerrar la simulación se puede apreciar
traj = np.array(trajectory)
print("\nRESULTADOS:")
print(f"Pasos: {len(traj)}  Distancia: {np.sum(np.linalg.norm(np.diff(traj,axis=0),axis=1)):.2f} m")
print(f"APF:        {avance_apf:.2f} m")
print(f"Preventivo:     {avance_preemptivo:.2f} m")
print(f"Escape lateral:  {avance_lat:.2f} m")
