import cv2
import numpy as np
import os

# Flow rates and particle speed scaling (μL/min)
flow_rates = [0, 5, 50, 100, 150, 400]
speed_map = {0: 0.0, 5: 0.1, 50: 0.3, 100: 0.6, 150: 1.0, 400: 1.8}

# Output folder
os.makedirs("data", exist_ok=True)

# Video properties
width, height = 640, 480
fps = 30
num_frames = 5  # short video ~0.5s

# Cluster grid
rows = 3
cols = 6
particles_per_cluster = 3

# Particle sizes: left → right gradient
min_radius = 4
max_radius = 12
column_radii = np.linspace(min_radius, max_radius, cols)  # left→right

# Horizontal and vertical spacing
h_spacing = width // (cols + 1)
v_spacing = height // (rows + 2)  # start near top

extra_sep = 2  # spacing inside cluster

# Movement direction (bottom-right)
angle = np.deg2rad(45)
dx, dy = np.cos(angle), np.sin(angle)

for flow in flow_rates:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"data/phantom_{flow}ul.avi", fourcc, fps, (width, height))

    cluster_x = []
    cluster_y = []
    particle_radii = []

    # Generate clusters row by row
    for row in range(rows):
        y = (row + 1) * v_spacing
        for col in range(cols):
            x = (col + 1) * h_spacing
            cluster_x.append(float(x))
            cluster_y.append(float(y))
            particle_radii.append(column_radii[col])  # left → right gradient applied per row

    cluster_x = np.array(cluster_x, dtype=float)
    cluster_y = np.array(cluster_y, dtype=float)
    particle_radii = np.array(particle_radii)

    # Offsets for particles (triangle with slight random variation)
    offsets_x = []
    offsets_y = []
    for r in particle_radii:
        sep = r*2 + extra_sep
        base_x = [-sep, 0, sep]
        base_y = [0, sep, 0]
        # Slight random variation
        base_x = [x + np.random.uniform(-r/2, r/2) for x in base_x]
        base_y = [y + np.random.uniform(-r/2, r/2) for y in base_y]
        offsets_x.append(base_x)
        offsets_y.append(base_y)
    offsets_x = np.array(offsets_x)
    offsets_y = np.array(offsets_y)

    # Speeds
    speed = speed_map[flow]
    vx = dx * speed
    vy = dy * speed

    for _ in range(num_frames):
        frame = np.random.normal(127, 10, (height, width)).astype(np.uint8)

        cluster_x += vx
        cluster_y += vy

        # Wrap around edges
        cluster_x = np.mod(cluster_x, width)
        cluster_y = np.mod(cluster_y, height)

        for i in range(len(cluster_x)):
            for j in range(particles_per_cluster):
                px = int(cluster_x[i] + offsets_x[i, j])
                py = int(cluster_y[i] + offsets_y[i, j])
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(frame, (px, py), int(particle_radii[i]), 255, -1)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()

print("✅ Generated top-aligned 3x6 grid with correct left→right size gradient across rows.")
