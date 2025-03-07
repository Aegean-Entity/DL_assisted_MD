import MDAnalysis as mda
import numpy as np
import torch

# Load the trajectory files
u = mda.Universe('ref.psf', 'out_eq51.dcd')

# Load the velocity and force DCD files
u.load_new('out_eq51.veldcd', format='DCD', velocities=True)
u.load_new('out_eq51.frcdcd', format='DCD', forces=True)



# Inspect the force DCD file
frc_reader = mda.coordinates.DCD.DCDReader('out_eq51.frcdcd')
print("Force DCD file header:", frc_reader.header)

# Select the potassium ion with resid 1
pot_ion = u.select_atoms('name POT and resid 1')

# Initialize arrays to store coordinates, velocities, and forces
n_frames = len(u.trajectory)
coordinates = np.zeros((n_frames, 3))  # 3D coordinates
velocities = np.zeros((n_frames, 3))   # 3D velocities
forces = np.zeros((n_frames, 3))       # 3D forces

# Iterate through trajectory frames and extract data
for i, ts in enumerate(u.trajectory):
    # Extract coordinates (positions in Å)
    coordinates[i] = pot_ion.positions[0]
    
    # Extract velocities (if available)
    if ts.has_velocities:
        velocities[i] = pot_ion.velocities[0]
    else:
        print(f"No velocities in frame {i}")
    
    # Extract forces (if available)
    if ts.has_forces:
        forces[i] = pot_ion.forces[0]
    else:
        print(f"No forces in frame {i}")

# Save as NumPy arrays
np.save('pot_ion_coordinates.npy', coordinates)
np.save('pot_ion_velocities.npy', velocities)
np.save('pot_ion_forces.npy', forces)

# Convert to PyTorch tensors and save
coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)
velocities_tensor = torch.tensor(velocities, dtype=torch.float32)
forces_tensor = torch.tensor(forces, dtype=torch.float32)

# Save the PyTorch tensors
torch.save(coordinates_tensor, 'pot_ion_coordinates.pt')
torch.save(velocities_tensor, 'pot_ion_velocities.pt')
torch.save(forces_tensor, 'pot_ion_forces.pt')

# Print data shape information
print(f"Data extracted for {n_frames} frames")
print(f"Coordinates shape: {coordinates.shape}")
print(f"Velocities shape: {velocities.shape}")
print(f"Forces shape: {forces.shape}")

# Combine all data into a single numpy array or tensor if needed
combined_data_np = np.column_stack((coordinates, velocities, forces))
combined_data_torch = torch.tensor(combined_data_np, dtype=torch.float32)

np.save('pot_ion_combined_data.npy', combined_data_np)
torch.save(combined_data_torch, 'pot_ion_combined_data.pt')