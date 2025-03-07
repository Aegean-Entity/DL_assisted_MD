import mdtraj as md
import numpy as np
import torch
import pandas as pd
import os

# Load the topology and trajectory files
topology = md.load_psf('ref.psf')
trajectory = md.load_dcd('out_eq50.dcd', top=topology)

# Get the number of frames
n_frames = trajectory.n_frames

# Debug output to understand the potassium atoms in the system
pot_atoms = [atom for atom in topology.atoms if 'POT' in atom.name]
print("Potassium atoms found:", [(atom.index, atom.name, atom.residue.resSeq) for atom in pot_atoms])

# Select potassium ions - using more reliable selection approach
ion_indices = topology.select("name POT")
print(f"Found {len(ion_indices)} potassium ions with indices: {ion_indices}")

# Now get the first potassium ion (equivalent to resid 1)
first_pot_index = ion_indices[0]
ion_indices = [first_pot_index]
print(f"Selected first potassium ion with index: {first_pot_index}")

# Initialize arrays to store coordinates
coordinates = np.zeros((n_frames, 3))  # 3D coordinates

# Extract coordinates from trajectory
for i in range(n_frames):
    # Extract positions (in nm, mdtraj default)
    coordinates[i] = trajectory.xyz[i, ion_indices[0]] * 10  # Convert to Angstroms

# Function to load DCD file as raw coordinates
def load_custom_dcd(filename, top, n_atoms):
    """Load a DCD file as raw coordinate data when standard loading doesn't work."""
    import struct
    import os
    
    n_frames = int(os.path.getsize(filename) / (4 + n_atoms * 3 * 4 + 4))
    data = np.zeros((n_frames, n_atoms, 3))
    
    with open(filename, 'rb') as f:
        # Skip the header
        f.seek(100)  # Skip initial part of header
        
        # Read the header info
        n_frames_file = struct.unpack('i', f.read(4))[0]
        
        # Skip to the data
        f.seek(260)  # Typical DCD header size
        
        # Read each frame
        for i in range(n_frames):
            # Read x, y, z coordinates separately (this is how DCD files are structured)
            for d in range(3):
                # Skip size indicator
                f.read(4)
                
                # Read coordinates for this dimension
                coords = np.frombuffer(f.read(4 * n_atoms), dtype=np.float32)
                data[i, :, d] = coords
                
                # Skip size indicator
                f.read(4)
    
    return data

# Get total number of atoms for loading velocity and force DCDs
n_atoms = topology.n_atoms

# Load velocity DCD (will be in Å/ps)
try:
    # Try loading with mdtraj first
    vel_traj = md.load_dcd('out_eq50.veldcd', top=topology)
    velocities = np.zeros((n_frames, 3))
    for i in range(n_frames):
        velocities[i] = vel_traj.xyz[i, ion_indices[0]] * 10  # Convert to Angstroms if needed
except Exception as e:
    print(f"Error loading velocity DCD with mdtraj: {e}")
    print("Attempting to load velocity DCD as raw data...")
    vel_data = load_custom_dcd('out_eq50.veldcd', topology, n_atoms)
    velocities = vel_data[:, ion_indices[0], :]

# Load force DCD (will be in kcal/mol/Å)
try:
    # Try loading with mdtraj first
    force_traj = md.load_dcd('out_eq50.frcdcd', top=topology)
    forces = np.zeros((n_frames, 3))
    for i in range(n_frames):
        forces[i] = force_traj.xyz[i, ion_indices[0]] * 10  # Convert to Angstroms if needed
except Exception as e:
    print(f"Error loading force DCD with mdtraj: {e}")
    print("Attempting to load force DCD as raw data...")
    force_data = load_custom_dcd('out_eq50.frcdcd', topology, n_atoms)
    forces = force_data[:, ion_indices[0], :]

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

# Save as CSV files
# Create DataFrames with appropriate column names
# Coordinates CSV
coord_df = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
coord_df.index.name = 'frame'
coord_df.to_csv('pot_ion_coordinates.csv')

# Velocities CSV
vel_df = pd.DataFrame(velocities, columns=['vx', 'vy', 'vz'])
vel_df.index.name = 'frame'
vel_df.to_csv('pot_ion_velocities.csv')

# Forces CSV
force_df = pd.DataFrame(forces, columns=['fx', 'fy', 'fz'])
force_df.index.name = 'frame'
force_df.to_csv('pot_ion_forces.csv')

# Combined data CSV
combined_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']
combined_df = pd.DataFrame(
    np.hstack((coordinates, velocities, forces)), 
    columns=combined_columns
)
combined_df.index.name = 'frame'
combined_df.to_csv('pot_ion_combined_data.csv')

# Print data shape information
print(f"Data extracted for {n_frames} frames")
print(f"Coordinates shape: {coordinates.shape}")
print(f"Velocities shape: {velocities.shape}")
print(f"Forces shape: {forces.shape}")

# Also extract protein Cα coordinates for the SchNet model
ca_indices = topology.select("name CA")
n_ca_atoms = len(ca_indices)
print(f"Found {n_ca_atoms} Cα atoms in the protein")

# Extract protein Cα coordinates from the first frame (assuming rigid protein)
# If the protein is flexible, you might want to use all frames
protein_ca_coords = trajectory.xyz[0, ca_indices, :] * 10  # Convert to Angstroms

# Save protein Cα coordinates for use in the SchNet model
np.save('protein_ca_coords.npy', protein_ca_coords)

# Save protein Cα coordinates as CSV
ca_coords_df = pd.DataFrame(
    protein_ca_coords,
    columns=['x', 'y', 'z']
)
ca_coords_df.index.name = 'atom_index'
ca_coords_df.to_csv('protein_ca_coords.csv')

# You can also combine all data into a single numpy array or tensor if needed
combined_data_np = np.column_stack((coordinates, velocities, forces))
combined_data_torch = torch.tensor(combined_data_np, dtype=torch.float32)

np.save('pot_ion_combined_data.npy', combined_data_np)
torch.save(combined_data_torch, 'pot_ion_combined_data.pt')

print("All data extracted and saved successfully in NPY, PT, and CSV formats!")