import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.affinity import scale

# Función para dibujar las bolas en la mesa de billar
def draw_balls(ax, balls_positions):
    for ball_name, ball_info in balls_positions.items():
        color = 'white' if ball_name == 'White' else 'grey'
        circle = plt.Circle(ball_info['position'], ball_info['radius'], color=color, label=ball_name)
        ax.add_artist(circle)
        plt.text(*ball_info['position'], ball_name, ha='center', va='center')

# Función para verificar si el camino está despejado
def is_path_clear(white_ball, target_ball, balls_positions):
    line = LineString([white_ball, target_ball])
    line = scale(line, 0.99, 0.99, origin=target_ball)
    for name, data in balls_positions.items():
        if name != 'White':
            ball = Point(data['position'])
            ball = ball.buffer(data['radius'])
            if line.intersects(ball):
                return False
    return True

# Función para analizar las trayectorias y encontrar la despejada
def analyze_clear_path(balls_positions):
    white_ball_position = balls_positions['White']['position']
    for ball_name, ball_info in balls_positions.items():
        if ball_name != 'White':
            if is_path_clear(white_ball_position, ball_info['position'], balls_positions):
                return ball_name
    return None

# Configuración de la mesa de billar y las bolas
# (deberás actualizar esto con las nuevas posiciones y radios de las bolas)
balls_positions = {
    'White': {'position': (980, 458), 'radius': 32},
    # Agregar las demás bolas aquí...
}

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 1796)  # Ancho estimado de la mesa
ax.set_ylim(0, 920)  # Alto estimado de la mesa
ax.set_aspect('equal')
ax.set_facecolor('green')

# Dibujar las bolas en la mesa
draw_balls(ax, balls_positions)

# Analizar y encontrar la trayectoria despejada
clear_shot = analyze_clear_path(balls_positions)

# Si se encuentra una trayectoria despejada, dibujarla
if clear_shot:
    plt.plot([balls_positions['White']['position'][0], balls_positions[clear_shot]['position'][0]], 
             [balls_positions['White']['position'][1], balls_positions[clear_shot]['position'][1]], 'y-')

# Invertir eje Y para coincidir con la orientación de la imagen
plt.gca().invert_yaxis()
plt.show()

# Imprimir el nombre de la bola con la trayectoria despejada
print(f"La trayectoria despejada es hacia la {clear_shot}." if clear_shot else "No hay trayectoria despejada.")
