import sympy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Generate Primes and Spiral Coordinates ---

def get_primes_and_coords(limit):
    """Generates primes up to limit and calculates their Ulam spiral coordinates."""
    primes = list(sympy.primerange(1, limit + 1))
    prime_set = set(primes)

    coords = {}
    x, y = 0, 0
    dx, dy = 1, 0  # Start moving right
    steps_to_turn = 1
    steps_taken = 0
    turns_made = 0

    for n in range(1, limit + 1):
        coords[n] = (x, y)
        if n in prime_set:
            pass # Store coord, already done

        # Move
        x += dx
        y += dy
        steps_taken += 1

        # Check for turn
        if steps_taken == steps_to_turn:
            steps_taken = 0
            # Rotate direction (dx, dy) -> (-dy, dx)
            dx, dy = -dy, dx
            turns_made += 1
            if turns_made % 2 == 0:
                steps_to_turn += 1 # Increase steps every 2 turns

    prime_coords = {p: coords[p] for p in primes if p in coords}
    return prime_coords

# Generate primes up to 10000
limit = 10000
prime_coordinates = get_primes_and_coords(limit)

# Extract x, y for plotting
prime_x = [coord[0] for coord in prime_coordinates.values()]
prime_y = [coord[1] for coord in prime_coordinates.values()]
prime_numbers = list(prime_coordinates.keys())

# --- 2. Define Simple Cymatic Pattern (Concentric Circles) ---
# These radii roughly correspond to the example image input_file_3.png visually
# In reality, these radii depend on solutions to wave equations (e.g., Bessel function roots for a circular plate)
nodal_radii = [30.0, 59.9, 85.0] # Approximate radii based on visual inspection of input_file_3.png

# --- 3. Identify Primes Near Nodal Lines ---
threshold = 2.0 # How close a prime needs to be to a radius to be considered "near"
primes_near_nodal = {} # Dict: radius -> list of primes near it

prime_radii = np.sqrt(np.array(prime_x)**2 + np.array(prime_y)**2)

nearby_prime_indices = []
nearby_prime_coords = []
nearby_prime_radii_map = {} # Store which radius each nearby prime corresponds to

for i, p_rad in enumerate(prime_radii):
    for nodal_r in nodal_radii:
        if abs(p_rad - nodal_r) <= threshold:
            nearby_prime_indices.append(i)
            nearby_prime_coords.append((prime_x[i], prime_y[i]))
            nearby_prime_radii_map[(prime_x[i], prime_y[i])] = nodal_r
            prime_num = prime_numbers[i]
            if nodal_r not in primes_near_nodal:
                primes_near_nodal[nodal_r] = []
            primes_near_nodal[nodal_r].append(prime_num)
            break # Assign prime to the first radius it's close to

nearby_prime_x = [coord[0] for coord in nearby_prime_coords]
nearby_prime_y = [coord[1] for coord in nearby_prime_coords]

# --- 4. Visualize ---
sns.set_style("darkgrid")
plt.figure(figsize=(10, 10))

# Plot all primes
plt.scatter(prime_x, prime_y, s=1, c='blue', label=f'Primes up to {limit}')

# Plot nodal circles
theta = np.linspace(0, 2 * np.pi, 200)
for i, r in enumerate(nodal_radii):
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    plt.plot(x_circle, y_circle, 'r-', label=f'Nodal Circle {i+1} (r={r:.1f})' if i < 1 else f'Nodal Circle {i+1}') # Only label one radius example for clarity

# Highlight primes near nodal lines
if nearby_prime_x:
    plt.scatter(nearby_prime_x, nearby_prime_y, s=80, facecolors='none', edgecolors='lime', linewidth=1.5,
                label=f'Primes near radii +/- {threshold}')

# Find plot limits based on max prime coord
max_coord = max(max(np.abs(prime_x)), max(np.abs(prime_y))) * 1.1
plt.xlim(-max_coord, max_coord)
plt.ylim(-max_coord, max_coord)

plt.title('Overlay of Prime Spiral and Simple Cymatic Pattern (Circles)')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='lower right')
plt.show()

# Print primes found near each nodal radius
print(f"Primes found near nodal radii (within +/- {threshold}):")
for r in sorted(primes_near_nodal.keys()):
    print(f"  Radius {r:.1f}: {sorted(primes_near_nodal[r])}")

# Calculate density for context (optional)
total_area = np.pi * (max_coord/1.1)**2
total_primes = len(prime_numbers)
avg_density = total_primes / total_area
print(f"\nAverage prime density in plotted area: {avg_density:.4f} primes per unit area")

for r in sorted(primes_near_nodal.keys()):
    annulus_area = np.pi * ((r + threshold)**2 - (r - threshold)**2)
    num_primes_found = len(primes_near_nodal[r])
    density_near_nodal = num_primes_found / annulus_area
    expected_primes = avg_density * annulus_area
    print(f"  Radius {r:.1f}: Found {num_primes_found} primes. Annulus Area={annulus_area:.1f}. Density={density_near_nodal:.4f}. Expected={expected_primes:.2f}")

