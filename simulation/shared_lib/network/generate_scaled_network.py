"""
Generate scaled network geometry for Storage Lanes network.
Target: Short=7.5km, Long=9.375km (1.25 ratio, development scale)
"""
import numpy as np

# Target path lengths (in meters)
SHORT_LENGTH = 7500   # 7.5 km
LONG_LENGTH = 9375    # 9.375 km (1.25x ratio)

# We'll use semicircular arcs for smooth curves
# For a semicircle: arc_length = pi * radius
# So radius = arc_length / pi

short_radius = SHORT_LENGTH / np.pi  # ~9549m
long_radius = LONG_LENGTH / np.pi    # ~14324m

print(f'Short path: {SHORT_LENGTH/1000:.0f}km, radius={short_radius:.1f}m')
print(f'Long path: {LONG_LENGTH/1000:.0f}km, radius={long_radius:.1f}m')
print(f'Ratio: {LONG_LENGTH/SHORT_LENGTH:.2f}')
print()

# Generate waypoints for semicircular arcs
# Using 100 points for smooth curves
n_points = 100

def generate_arc_points(radius, direction='up'):
    """Generate semicircular arc points.
    direction: 'up' = curves upward (positive Y), 'down' = curves downward (negative Y)
    """
    # Semicircle goes from 0 to pi radians
    # Start at (0,0) and end at (2*radius, 0)
    angles = np.linspace(np.pi, 0, n_points)  # from left to right
    
    # Center of semicircle is at (radius, 0)
    x = radius + radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    if direction == 'down':
        y = -y
    
    return x, y

# Generate forward short path (curves up)
x_short, y_short = generate_arc_points(short_radius, 'up')

# Generate forward long path (curves down)
x_long, y_long = generate_arc_points(long_radius, 'down')

# Format as SUMO shape string
def format_shape(x, y, x_offset=0, y_offset=0):
    coords = [f'{xi + x_offset:.2f},{yi + y_offset:.2f}' for xi, yi in zip(x, y)]
    return ' '.join(coords)

# Print key dimensions
print(f'Short path horizontal span: {2*short_radius/1000:.1f} km')
print(f'Long path horizontal span: {2*long_radius/1000:.1f} km')
print()

# Entry edge length (scaled proportionally)
entry_length = 750  # 0.75km entry (proportionally scaled)

print(f'Entry edge length: {entry_length/1000:.1f} km')
print()

# Calculate free-flow times at 40 m/s (144 km/h)
speed = 40.0
short_ff = SHORT_LENGTH / speed
long_ff = LONG_LENGTH / speed
print(f'Free-flow times at {speed} m/s:')
print(f'  Short: {short_ff:.0f}s ({short_ff/60:.1f} min)')
print(f'  Long: {long_ff:.0f}s ({long_ff/60:.1f} min)')

# Generate shape strings
short_fwd_shape = format_shape(x_short, y_short)
long_fwd_shape = format_shape(x_long, y_long)

# For backward paths, we reverse and mirror
# Start from the end point and go backwards
x_short_bwd = x_short[::-1]
y_short_bwd = y_short[::-1]
x_long_bwd = x_long[::-1]
y_long_bwd = y_long[::-1]

# Offset backward paths in Y to separate from forward
y_offset_bwd = 125  # 125m offset (proportionally scaled)

short_bwd_shape = format_shape(x_short_bwd, y_short_bwd, y_offset=y_offset_bwd)
long_bwd_shape = format_shape(x_long_bwd, y_long_bwd, y_offset=y_offset_bwd)

print()
print('='*60)
print('SHAPE STRINGS FOR edge.xml')
print('='*60)
print()
print('SHORT_FWD:')
print(short_fwd_shape[:200] + '...')
print()
print('LONG_FWD:')
print(long_fwd_shape[:200] + '...')

# Save to file for easy copy
with open('generated_shapes.txt', 'w') as f:
    f.write('SHORT_FWD:\n')
    f.write(short_fwd_shape + '\n\n')
    f.write('LONG_FWD:\n')
    f.write(long_fwd_shape + '\n\n')
    f.write('SHORT_BWD:\n')
    f.write(short_bwd_shape + '\n\n')
    f.write('LONG_BWD:\n')
    f.write(long_bwd_shape + '\n\n')

print()
print('Shapes saved to generated_shapes.txt')
