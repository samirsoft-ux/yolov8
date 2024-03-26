import matplotlib.pyplot as plt
import numpy as np

# Coordenadas de las bolas y buchacas
bola_blanca = (980, 457)
bolas = {
    2: {'pos': (616, 351), 'radio': 34},
    3: {'pos': (966, 303), 'radio': 32},
    4: {'pos': (1001, 587), 'radio': 31},
    5: {'pos': (325, 277), 'radio': 36},
    6: {'pos': (1406, 559), 'radio': 32},
    7: {'pos': (1006, 695), 'radio': 31}
}

buchacas = {
    'esq_sup_izq': (177, 165),
    'esq_sup_der': (1771, 141),
    'esq_inf_der': (1796, 865),
    'esq_inf_izq': (186, 920),
    'mitad_sup': (974, 153),
    'mitad_inf': (991, 892)
}

# Inicializar la figura y los ejes
fig, ax = plt.subplots(figsize=(16, 8))

# Dibujar la mesa de billar
for pos in bolas.values():
    ax.add_patch(plt.Circle(pos['pos'], pos['radio'], color='blue', fill=False))
ax.add_patch(plt.Circle(bola_blanca, 32, color='red', fill=False))

# Dibujar las trayectorias de las bolas hacia las buchacas más cercanas y desde la bola blanca
intersecciones = {}
for id_bola, bola in bolas.items():
    # Encontrar la buchaca más cercana para cada bola
    distancias = {k: np.linalg.norm(np.array(v) - np.array(bola['pos'])) for k, v in buchacas.items()}
    buchaca_cercana = min(distancias, key=distancias.get)
    
    # Coordenadas de la buchaca más cercana
    buchaca_pos = buchacas[buchaca_cercana]

    # Calcular el punto medio hacia la buchaca
    punto_medio_buchaca = np.array(buchaca_pos) * 0.5 + np.array(bola['pos']) * 0.5

    # Dibujar la trayectoria desde la bola hacia la buchaca
    ax.plot([bola['pos'][0], punto_medio_buchaca[0]], [bola['pos'][1], punto_medio_buchaca[1]], 'g--')
    
    # Dibujar la trayectoria desde la bola blanca hacia la bola
    ax.plot([bola_blanca[0], bola['pos'][0]], [bola_blanca[1], bola['pos'][1]], 'r--')

    # Calcular la intersección de las trayectorias
    # Esta intersección se produce si el punto medio de la buchaca está en la línea entre la bola blanca y la bola
    direccion_bola_blanca_a_bola = (np.array(bola['pos']) - np.array(bola_blanca))
    direccion_bola_a_buchaca = (np.array(punto_medio_buchaca) - np.array(bola['pos']))
    
    # Normalizar las direcciones
    dir_bb_a_bola_norm = direccion_bola_blanca_a_bola / np.linalg.norm(direccion_bola_blanca_a_bola)
    dir_bola_a_buchaca_norm = direccion_bola_a_buchaca / np.linalg.norm(direccion_bola_a_buchaca)
    
    # Si las direcciones normalizadas son iguales o casi iguales, se considera que hay intersección
    if np.allclose(dir_bb_a_bola_norm, dir_bola_a_buchaca_norm, atol=1e-03):
        intersecciones[id_bola] = bola['pos']

# Ajustar límites
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)

# Invertir el eje y para que coincida con la imagen subida
ax.invert_yaxis()

# Desactivar los ejes
ax.axis('off')

# Mostrar la figura
plt.show()

# Devolver las intersecciones
intersecciones

#------------------------------------------------------------------
def check_line_obstruction(start, end, balls, radius_threshold):
    """
    Check if the line from start to end is obstructed by any ball within the balls dictionary.

    Parameters:
    - start: Starting point (x, y) of the line.
    - end: Ending point (x, y) of the line.
    - balls: Dictionary of balls with their positions and radii.
    - radius_threshold: The radius to consider for obstruction.

    Returns:
    - obstructed: Boolean indicating if the line is obstructed.
    - obstructing_balls: List of ball IDs that are obstructing the line.
    """
    # Vector from start to end
    line_vector = np.array(end) - np.array(start)
    # Normalized direction vector
    line_direction = line_vector / np.linalg.norm(line_vector)
    # List of balls that obstruct the line
    obstructing_balls = []

    for id_bola, bola_info in balls.items():
        # Position of the current ball
        ball_pos = np.array(bola_info['pos'])
        # Vector from start point to the center of the current ball
        start_to_ball_vector = ball_pos - np.array(start)
        # Projection of the start_to_ball_vector onto the line_direction
        projection_length = np.dot(start_to_ball_vector, line_direction)
        # The projection point on the line
        projection_point = np.array(start) + line_direction * projection_length
        # Distance from the projection point to the ball's center
        closest_distance = np.linalg.norm(projection_point - ball_pos)

        # Check if the distance is less than the threshold and the projection is between start and end
        if (closest_distance <= radius_threshold + bola_info['radio']) and (0 <= projection_length <= np.linalg.norm(line_vector)):
            obstructing_balls.append(id_bola)

    # The line is obstructed if there are any obstructing balls
    obstructed = len(obstructing_balls) > 0

    return obstructed, obstructing_balls


# Revisar las trayectorias de las bolas hacia las buchacas para obstrucciones
trayectorias_posibles = {}
for id_bola, bola in bolas.items():
    # Encontrar la buchaca más cercana para cada bola
    distancias = {k: np.linalg.norm(np.array(v) - np.array(bola['pos'])) for k, v in buchacas.items()}
    buchaca_cercana = min(distancias, key=distancias.get)
    
    # Coordenadas de la buchaca más cercana
    buchaca_pos = buchacas[buchaca_cercana]

    # Calcular el punto medio hacia la buchaca
    punto_medio_buchaca = np.array(buchaca_pos) * 0.5 + np.array(bola['pos']) * 0.5

    # Dibujar la trayectoria desde la bola hacia la buchaca
    ax.plot([bola['pos'][0], punto_medio_buchaca[0]], [bola['pos'][1], punto_medio_buchaca[1]], 'g--')

    # Revisar si la trayectoria está obstruida
    bolas_para_revisar = {k: v for k, v in bolas.items() if k != id_bola}  # Excluir la bola actual de la revisión
    obstruida, bolas_obstructoras = check_line_obstruction(bola['pos'], buchaca_pos, bolas_para_revisar, bola['radio'])

    if not obstruida:
        # Si la trayectoria no está obstruida, se añade a las trayectorias posibles
        trayectorias_posibles[id_bola] = buchaca_cercana

# Mostrar las trayectorias posibles y las bolas obstructoras
plt.show()
trayectorias_posibles

#------------------------------------------------------------------------------------------------
# Inicializar la figura y los ejes para dibujar los resultados
fig, ax = plt.subplots(figsize=(16, 8))

# Dibujar la mesa de billar
for pos in bolas.values():
    ax.add_patch(plt.Circle(pos['pos'], pos['radio'], color='blue', fill=False))
ax.add_patch(plt.Circle(bola_blanca, 32, color='red', fill=False))

# Dibujar las trayectorias de las bolas hacia las buchacas más cercanas y desde la bola blanca
for id_bola, bola in bolas.items():
    # Encontrar la buchaca más cercana para cada bola
    distancias = {k: np.linalg.norm(np.array(v) - np.array(bola['pos'])) for k, v in buchacas.items()}
    buchaca_cercana = min(distancias, key=distancias.get)
    
    # Coordenadas de la buchaca más cercana
    buchaca_pos = buchacas[buchaca_cercana]

    # Calcular el punto medio hacia la buchaca
    punto_medio_buchaca = np.array(buchaca_pos) * 0.5 + np.array(bola['pos']) * 0.5

    # Dibujar la trayectoria desde la bola hacia la buchaca si la bola tiene un tiro posible
    if id_bola in trayectorias_posibles:
        ax.plot([bola['pos'][0], buchaca_pos[0]], [bola['pos'][1], buchaca_pos[1]], 'g-')

    # Dibujar la trayectoria desde la bola blanca hacia la bola
    ax.plot([bola_blanca[0], bola['pos'][0]], [bola_blanca[1], bola['pos'][1]], 'r--')

# Ajustar límites
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)

# Invertir el eje y para que coincida con la imagen subida
ax.invert_yaxis()

# Desactivar los ejes
ax.axis('off')

# Mostrar la figura
plt.show()

#-------------------------------------------------------
# Función para verificar obstrucciones en la trayectoria roja
def check_red_line_obstruction(start, end, balls, radius_threshold):
    """
    Check if the line from start to end (red line) is obstructed by any ball within the balls dictionary.

    Parameters:
    - start: Starting point (x, y) of the line (bola blanca).
    - end: Ending point (x, y) of the line (otra bola).
    - balls: Dictionary of balls with their positions and radii.
    - radius_threshold: The radius to consider for obstruction.

    Returns:
    - obstructed: Boolean indicating if the line is obstructed.
    """
    # Vector from start to end
    line_vector = np.array(end) - np.array(start)
    # Normalized direction vector
    line_direction = line_vector / np.linalg.norm(line_vector)

    for id_bola, bola_info in balls.items():
        if np.array_equal(bola_info['pos'], end):
            # Skip the end ball (as it is the destination of the red line)
            continue

        # Position of the current ball
        ball_pos = np.array(bola_info['pos'])
        # Vector from start point to the center of the current ball
        start_to_ball_vector = ball_pos - np.array(start)
        # Projection of the start_to_ball_vector onto the line_direction
        projection_length = np.dot(start_to_ball_vector, line_direction)
        # The projection point on the line
        projection_point = np.array(start) + line_direction * projection_length
        # Distance from the projection point to the ball's center
        closest_distance = np.linalg.norm(projection_point - ball_pos)

        # Check if the distance is less than the threshold and the projection is between start and end
        if (closest_distance <= radius_threshold + bola_info['radio']) and (0 <= projection_length <= np.linalg.norm(line_vector)):
            # If the line is obstructed, we return True
            return True

    # If no obstruction was found, return False
    return False

# Revisar las trayectorias de las bolas hacia las buchacas para obstrucciones incluyendo la trayectoria roja
trayectorias_finales_posibles = {}
for id_bola, bola in bolas.items():
    if id_bola in trayectorias_posibles:
        # Si la trayectoria verde no está obstruida, revisamos la roja
        if not check_red_line_obstruction(bola_blanca, bola['pos'], bolas, bola['radio']):
            # Si la trayectoria roja tampoco está obstruida, se añade a las trayectorias finales posibles
            trayectorias_finales_posibles[id_bola] = trayectorias_posibles[id_bola]

# Inicializar la figura y los ejes para dibujar los resultados finales
fig, ax = plt.subplots(figsize=(16, 8))

# Dibujar la mesa de billar
for pos in bolas.values():
    ax.add_patch(plt.Circle(pos['pos'], pos['radio'], color='blue', fill=False))
ax.add_patch(plt.Circle(bola_blanca, 32, color='red', fill=False))

# Dibujar las trayectorias finales posibles
for id_bola, bola in bolas.items():
    # Encontrar la buchaca más cercana para cada bola
    distancias = {k: np.linalg.norm(np.array(v) - np.array(bola['pos'])) for k, v in buchacas.items()}
    buchaca_cercana = min(distancias, key=distancias.get)
    buchaca_pos = buchacas[buchaca_cercana]

    if id_bola in trayectorias_finales_posibles:
        # Dibujar la trayectoria verde final (desde la bola hacia la buchaca)
        ax.plot([bola['pos'][0], buchaca_pos[0]], [bola['pos'][1], buchaca_pos[1]], 'g-')
        # Dibujar la trayectoria roja final (desde la bola blanca hacia la bola)
        ax.plot([bola_blanca[0], bola['pos'][0]], [bola_blanca[1], bola['pos'][1]], 'r-')

# Ajustar límites
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)

# Invertir el eje y para que coincida con la imagen subida
ax.invert_yaxis()

# Desactivar los ejes
ax.axis('off')

# Mostrar la figura
plt.show()

# Las trayectorias finales posibles sin

#----------------------------------------------------------------------------------------
Código completo

import matplotlib.pyplot as plt
import numpy as np

def check_line_obstruction(start, end, balls, radius_threshold):
    # Vector from start to end
    line_vector = np.array(end) - np.array(start)
    # Normalized direction vector
    line_direction = line_vector / np.linalg.norm(line_vector)
    # List of balls that obstruct the line
    obstructing_balls = []

    for id_bola, bola_info in balls.items():
        # Position of the current ball
        ball_pos = np.array(bola_info['pos'])
        # Vector from start point to the center of the current ball
        start_to_ball_vector = ball_pos - np.array(start)
        # Projection of the start_to_ball_vector onto the line_direction
        projection_length = np.dot(start_to_ball_vector, line_direction)
        # The projection point on the line
        projection_point = np.array(start) + line_direction * projection_length
        # Distance from the projection point to the ball's center
        closest_distance = np.linalg.norm(projection_point - ball_pos)

        # Check if the distance is less than the threshold and the projection is between start and end
        if (closest_distance <= radius_threshold + bola_info['radio']) and (0 <= projection_length <= np.linalg.norm(line_vector)):
            obstructing_balls.append(id_bola)

    # The line is obstructed if there are any obstructing balls
    obstructed = len(obstructing_balls) > 0

    return obstructed, obstructing_balls

def check_red_line_obstruction(start, end, balls, radius_threshold):
    # Vector from start to end
    line_vector = np.array(end) - np.array(start)
    # Normalized direction vector
    line_direction = line_vector / np.linalg.norm(line_vector)

    for id_bola, bola_info in balls.items():
        if np.array_equal(bola_info['pos'], end):
            # Skip the end ball (as it is the destination of the red line)
            continue

        # Position of the current ball
        ball_pos = np.array(bola_info['pos'])
        # Vector from start point to the center of the current ball
        start_to_ball_vector = ball_pos - np.array(start)
        # Projection of the start_to_ball_vector onto the line_direction
        projection_length = np.dot(start_to_ball_vector, line_direction)
        # The projection point on the line
        projection_point = np.array(start) + line_direction * projection_length
        # Distance from the projection point to the ball's center
        closest_distance = np.linalg.norm(projection_point - ball_pos)

        # Check if the distance is less than the threshold and the projection is between start and end
        if (closest_distance <= radius_threshold + bola_info['radio']) and (0 <= projection_length <= np.linalg.norm(line_vector)):
            # If the line is obstructed, we return True
            return True

    # If no obstruction was found, return False
    return False

# Coordenadas de las bolas y buchacas (deberás actualizar estas con tus propios datos)
bola_blanca = (980, 457)
bolas = {
    # ... (resto de las bolas)
}

buchacas = {
    # ... (resto de las buchacas)
}

# Revisar las trayectorias de las bolas hacia las buchacas para obstrucciones incluyendo la trayectoria roja
trayectorias_finales_posibles = {}
for id_bola, bola in bolas.items():
    # ... (proceso para calcular trayectorias posibles y verificar obstrucciones)

# Inicializar la figura y los ejes para dibujar los resultados finales
fig, ax = plt.subplots(figsize=(16, 8))
# ... (código para dibujar la mesa, las bolas, y las trayectorias)
# Mostrar la figura
plt.show()
