import numpy as np
import matplotlib.pyplot as plt
import sympy

# Function to generate primes up to N
def get_primes(n_max):
    primes = list(sympy.primerange(1, n_max + 1))
    return primes

# Function to generate spiral coordinates for primes
def generate_prime_spiral_coords(primes, scale_factor=1.0, angle_factor=2*np.pi / ((1+np.sqrt(5))/2)): # Using golden angle for spacing
    coords = []
    for p in primes:
        # Using a Vogel-like spiral: r = sqrt(n), theta related to sqrt(n)
        # Or simpler: r = p, theta = p
        # Let's try Vogel spiral for better spacing aesthetics often seen in nature/plots
        r = scale_factor * np.sqrt(p)
        theta = p * angle_factor # Angle proportional to the number itself
        # Convert polar to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        coords.append((x, y))
    return np.array(coords)

# Function to generate points for simple cymatic patterns (concentric circles)
def generate_cymatic_circles(radii, n_points=200):
    patterns = []
    for r in radii:
        thetas = np.linspace(0, 2 * np.pi, n_points)
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        patterns.append(np.stack([x, y], axis=-1))
    return patterns

# --- Parameters ---
N_max = 10000  # Max number for prime spiral (adjust as needed)
spiral_scale = 1.0 # Scale factor for spiral radius

# --- Generate Data ---
primes = get_primes(N_max)
prime_coords = generate_prime_spiral_coords(primes, scale_factor=spiral_scale)

# Determine max radius for cymatic patterns based on spiral extent
max_r_prime = np.max(np.sqrt(prime_coords[:, 0]**2 + prime_coords[:, 1]**2))

# Example cymatic pattern: 3 concentric circles within the spiral's range
cymatic_radii = [max_r_prime * 0.3, max_r_prime * 0.6, max_r_prime * 0.9]
cymatic_patterns = generate_cymatic_circles(cymatic_radii)

# --- Plotting ---
plt.style.use('seaborn-v0_8-darkgrid') # Nicer style
fig, ax = plt.subplots(figsize=(10, 10))

# Plot prime spiral points
ax.scatter(prime_coords[:, 0], prime_coords[:, 1], s=2, c='blue', alpha=0.7, label=f'Primes up to {N_max}')

# Plot cymatic pattern lines
for i, pattern in enumerate(cymatic_patterns):
    ax.plot(pattern[:, 0], pattern[:, 1], color='red', linestyle='-', linewidth=1.5, alpha=0.8, label=f'Nodal Circle {i+1}' if i==0 else None)

# --- Conceptual Encoding Example ---
# Let's find primes "close" to the middle circle (radius = cymatic_radii[1])
target_radius = cymatic_radii[1]
tolerance = max_r_prime * 0.02 # Define "close" as within 2% of max radius

prime_radii = np.sqrt(prime_coords[:, 0]**2 + prime_coords[:, 1]**2)
close_indices = np.where(np.abs(prime_radii - target_radius) < tolerance)[0]
close_primes = [primes[i] for i in close_indices]
close_coords = prime_coords[close_indices]

# Highlight the "encoded" primes
ax.scatter(close_coords[:, 0], close_coords[:, 1], s=50, facecolors='none', edgecolors='lime', linewidth=1.5, label=f'Primes near r={target_radius:.1f} (+/- {tolerance:.1f})')

print(f"Conceptual 'Encoding': Primes close to the middle circle (r={target_radius:.2f}):")
print(close_primes)


# --- Final Plot Adjustments ---
ax.set_title('Overlay of Prime Spiral and Simple Cymatic Pattern (Circles)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.grid(True)
plt.show()