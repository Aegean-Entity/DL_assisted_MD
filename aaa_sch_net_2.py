import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class RadialBasisFunctions(nn.Module):
    """
    Radial Basis Function expansion for distances.
    """
    def __init__(self, n_rbf=50, cutoff=10.0, trainable=False):
        super(RadialBasisFunctions, self).__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        
        # Initialize centers evenly between 0 and cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.centers = nn.Parameter(centers, requires_grad=trainable)
        
        # Initialize widths relative to spacing of centers
        widths = torch.FloatTensor([0.5 * (cutoff / n_rbf)])
        self.widths = nn.Parameter(widths, requires_grad=trainable)
    
    def forward(self, distances):
        # Expand distances in RBF basis
        # distances shape: [batch_size, n_interactions]
        distances = distances.unsqueeze(-1)  # [batch_size, n_interactions, 1]
        
        # Compute RBF values
        rbf = torch.exp(-((distances - self.centers) ** 2) / (self.widths ** 2))
        
        # Apply cutoff
        envelope = self._cutoff_function(distances)
        return rbf * envelope
    
    def _cutoff_function(self, distances):
        # Ensures smooth decay to zero at cutoff
        # Cosine cutoff function
        envelope = torch.zeros_like(distances)
        mask = distances < self.cutoff
        envelope[mask] = 0.5 * (torch.cos(distances[mask] * np.pi / self.cutoff) + 1.0)
        return envelope

class InteractionBlock(nn.Module):
    """
    SchNet-inspired interaction block for computing messages between ion and protein atoms.
    """
    def __init__(self, hidden_dim=128, n_rbf=50, cutoff=10.0):
        super(InteractionBlock, self).__init__()
        
        self.rbf = RadialBasisFunctions(n_rbf=n_rbf, cutoff=cutoff)
        
        # Message passing network
        self.message_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Update network for ion features
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, ion_features, distances):
        # distances: [batch_size, n_protein_atoms]
        
        # Expand distances into RBF features
        rbf_features = self.rbf(distances)  # [batch_size, n_protein_atoms, n_rbf]
        
        # Compute messages
        messages = self.message_net(rbf_features)  # [batch_size, n_protein_atoms, hidden_dim]
        
        # Aggregate messages from all protein atoms to the ion
        aggr_messages = torch.sum(messages, dim=1)  # [batch_size, hidden_dim]
        
        # Update ion features
        ion_features = ion_features + self.update_net(aggr_messages)
        
        return ion_features

class TemporalBlock(nn.Module):
    """
    Processes temporal aspects of ion dynamics using LSTMs.
    """
    def __init__(self, hidden_dim=128, lstm_layers=2):
        super(TemporalBlock, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Project bidirectional output back to hidden_dim
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        x, _ = self.lstm(x)
        x = self.proj(x)
        return x

class PhysicsDecoder(nn.Module):
    """
    Decodes features into physical quantities (position, velocity, force) 
    with physics-informed constraints.
    """
    def __init__(self, hidden_dim=128):
        super(PhysicsDecoder, self).__init__()
        
        # Decoders for position, velocity, and force
        self.pos_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # 3D coordinates
        )
        
        self.vel_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # 3D velocity
        )
        
        self.force_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # 3D force
        )
        
        # Physics parameters (can be learned)
        self.dt = nn.Parameter(torch.tensor(0.001), requires_grad=False)  # Time step
        self.mass = nn.Parameter(torch.tensor(39.1), requires_grad=False)  # Mass of K+ ion in atomic units
        
    def forward(self, x, enforce_physics=True):
        # Decode raw outputs
        pos_pred = self.pos_decoder(x)
        vel_pred = self.vel_decoder(x)
        force_pred = self.force_decoder(x)
        
        if enforce_physics:
            # Apply physics constraints - enforce F = ma
            # Compute acceleration from position using finite difference
            # For middle frames
            batch_size, seq_len, _ = pos_pred.shape
            if seq_len > 2:
                pos_mid = pos_pred[:, 1:-1, :]
                pos_prev = pos_pred[:, :-2, :]
                pos_next = pos_pred[:, 2:, :]
                
                # Second derivative approximation (acceleration)
                accel = (pos_next - 2 * pos_mid + pos_prev) / (self.dt ** 2)
                
                # Force should match mass * acceleration (F = ma)
                force_physics = self.mass * accel
                
                # Update force predictions for middle frames
                force_pred[:, 1:-1, :] = force_physics
                
                # Update velocity to be consistent with positions
                vel_physics = (pos_next - pos_prev) / (2 * self.dt)
                vel_pred[:, 1:-1, :] = vel_physics
        
        return pos_pred, vel_pred, force_pred
    
    def enforce_energy_conservation(self, pos, vel, force, potential_energy_fn=None):
        """
        Adjust velocities to enforce energy conservation.
        """
        # Compute kinetic energy: 0.5 * m * v^2
        kinetic_energy = 0.5 * self.mass * torch.sum(vel ** 2, dim=-1)
        
        if potential_energy_fn is not None:
            # Compute potential energy using provided function
            potential_energy = potential_energy_fn(pos)
            
            # Total energy should be conserved
            total_energy = kinetic_energy + potential_energy
            
            # Compute mean total energy
            mean_energy = torch.mean(total_energy)
            
            # Adjust velocities to maintain constant energy
            energy_ratio = torch.sqrt(2 * (mean_energy - potential_energy) / (self.mass * torch.sum(vel ** 2, dim=-1)))
            energy_ratio = torch.clamp(energy_ratio, 0.5, 2.0)  # Limit adjustments
            
            # Apply adjustment to velocities
            vel_adjusted = vel * energy_ratio.unsqueeze(-1)
            
            return pos, vel_adjusted, force
        
        return pos, vel, force

class IonTransportDataset(Dataset):
    """
    Dataset for ion transport with protein context.
    """
    def __init__(self, ion_coords, ion_vels, ion_forces, protein_coords, seq_length=32, stride=1):
        # Ion dynamics data
        self.ion_coords = torch.tensor(ion_coords, dtype=torch.float32)
        self.ion_vels = torch.tensor(ion_vels, dtype=torch.float32)
        self.ion_forces = torch.tensor(ion_forces, dtype=torch.float32)
        
        # Protein coordinates (Cα atoms)
        self.protein_coords = torch.tensor(protein_coords, dtype=torch.float32)
        
        self.seq_length = seq_length
        self.stride = stride
        
        # Calculate number of sequences
        n_frames = len(ion_coords)
        self.n_sequences = max(0, (n_frames - seq_length) // stride + 1)
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length
        
        # Get sequence of ion dynamics
        ion_coords_seq = self.ion_coords[start_idx:end_idx]
        ion_vels_seq = self.ion_vels[start_idx:end_idx]
        ion_forces_seq = self.ion_forces[start_idx:end_idx]
        
        # Calculate distances to protein atoms for each frame
        distances = []
        for i in range(self.seq_length):
            # Calculate pairwise distances between ion and protein atoms
            frame_dists = torch.norm(ion_coords_seq[i].unsqueeze(0) - self.protein_coords, dim=1)
            distances.append(frame_dists)
        
        distances = torch.stack(distances)
        
        return {
            'coords': ion_coords_seq,
            'vels': ion_vels_seq,
            'forces': ion_forces_seq,
            'distances': distances
        }

class PhysicsInformedSchNet(nn.Module):
    """
    Complete physics-informed SchNet model for ion transport prediction.
    """
    def __init__(self, hidden_dim=128, n_interaction_blocks=3, n_rbf=50, cutoff=10.0, lstm_layers=2):
        super(PhysicsInformedSchNet, self).__init__()
        
        # Initial embedding of concatenated coordinates, velocities, and forces
        self.embedding = nn.Linear(9, hidden_dim)  # 3 (coords) + 3 (vels) + 3 (forces) = 9
        
        # Interaction blocks
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(hidden_dim=hidden_dim, n_rbf=n_rbf, cutoff=cutoff)
            for _ in range(n_interaction_blocks)
        ])
        
        # Temporal processing
        self.temporal_block = TemporalBlock(hidden_dim=hidden_dim, lstm_layers=lstm_layers)
        
        # Physics-aware decoder
        self.physics_decoder = PhysicsDecoder(hidden_dim=hidden_dim)
        
    def forward(self, coords, vels, forces, distances, enforce_physics=True, extrapolation_steps=0):
        batch_size, seq_len = coords.shape[0], coords.shape[1]
        
        # Concatenate input features
        x = torch.cat([coords, vels, forces], dim=-1)  # [batch_size, seq_len, 9]
        
        # Initial embedding
        h = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Process each timestep with interaction blocks
        h_sequence = []
        for t in range(seq_len):
            h_t = h[:, t]  # Features at timestep t
            dist_t = distances[:, t]  # Distances at timestep t
            
            # Apply interaction blocks
            for block in self.interaction_blocks:
                h_t = block(h_t, dist_t)
            
            h_sequence.append(h_t)
        
        # Stack processed features back into sequence
        h = torch.stack(h_sequence, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Apply temporal processing
        h = self.temporal_block(h)
        
        # Decode physical quantities
        pos_pred, vel_pred, force_pred = self.physics_decoder(h, enforce_physics=enforce_physics)
        
        # For extrapolation
        if extrapolation_steps > 0:
            extrapolated_pos, extrapolated_vel, extrapolated_force = self.extrapolate(
                pos_pred[:, -1], vel_pred[:, -1], force_pred[:, -1], 
                distances[:, -1], steps=extrapolation_steps
            )
            
            # Concatenate with predictions
            pos_pred = torch.cat([pos_pred, extrapolated_pos[:, 1:]], dim=1)
            vel_pred = torch.cat([vel_pred, extrapolated_vel[:, 1:]], dim=1)
            force_pred = torch.cat([force_pred, extrapolated_force[:, 1:]], dim=1)
        
        return pos_pred, vel_pred, force_pred
    
    def extrapolate(self, last_pos, last_vel, last_force, last_distances, steps=10):
        """
        Extrapolate future positions, velocities, and forces.
        """
        batch_size = last_pos.shape[0]
        dt = self.physics_decoder.dt.item()
        
        # Initialize arrays for extrapolated values
        extrapolated_pos = torch.zeros(batch_size, steps + 1, 3, device=last_pos.device)
        extrapolated_vel = torch.zeros(batch_size, steps + 1, 3, device=last_vel.device)
        extrapolated_force = torch.zeros(batch_size, steps + 1, 3, device=last_force.device)
        
        # Set initial values
        extrapolated_pos[:, 0] = last_pos
        extrapolated_vel[:, 0] = last_vel
        extrapolated_force[:, 0] = last_force
        
        # Extrapolate step by step
        for i in range(1, steps + 1):
            # Extract previous values
            prev_pos = extrapolated_pos[:, i-1]
            prev_vel = extrapolated_vel[:, i-1]
            prev_force = extrapolated_force[:, i-1]
            
            # Calculate new position using velocity verlet integration
            # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            accel = prev_force / self.physics_decoder.mass
            new_pos = prev_pos + prev_vel * dt + 0.5 * accel * dt**2
            
            # Concatenate with current features and distances
            curr_features = torch.cat([prev_pos, prev_vel, prev_force], dim=-1)
            curr_features = self.embedding(curr_features)
            
            # Update distances based on new position (approximation)
            # In a real implementation, you'd need to update distances based on protein positions
            
            # Process through interaction blocks
            for block in self.interaction_blocks:
                curr_features = block(curr_features, last_distances)
            
            # Decode to get new predictions
            curr_features = curr_features.unsqueeze(1)  # Add sequence dimension
            curr_features = self.temporal_block(curr_features).squeeze(1)
            
            # Decode with physics constraints
            _, new_vel_raw, new_force = self.physics_decoder(curr_features.unsqueeze(1), enforce_physics=False)
            new_vel_raw = new_vel_raw.squeeze(1)
            new_force = new_force.squeeze(1)
            
            # Update velocity using force
            # v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
            new_accel = new_force / self.physics_decoder.mass
            new_vel = prev_vel + 0.5 * (accel + new_accel) * dt
            
            # Store extrapolated values
            extrapolated_pos[:, i] = new_pos
            extrapolated_vel[:, i] = new_vel
            extrapolated_force[:, i] = new_force
        
        return extrapolated_pos, extrapolated_vel, extrapolated_force

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """
    Train the physics-informed SchNet model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Loss weights
    w_pos, w_vel, w_force = 1.0, 0.5, 0.1
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # Move data to device
            coords = batch['coords'].to(device)
            vels = batch['vels'].to(device)
            forces = batch['forces'].to(device)
            distances = batch['distances'].to(device)
            
            # Forward pass
            pos_pred, vel_pred, force_pred = model(coords, vels, forces, distances)
            
            # Calculate losses
            pos_loss = F.mse_loss(pos_pred, coords)
            vel_loss = F.mse_loss(vel_pred, vels)
            force_loss = F.mse_loss(force_pred, forces)
            
            # Combined loss
            loss = w_pos * pos_loss + w_vel * vel_loss + w_force * force_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                coords = batch['coords'].to(device)
                vels = batch['vels'].to(device)
                forces = batch['forces'].to(device)
                distances = batch['distances'].to(device)
                
                # Forward pass
                pos_pred, vel_pred, force_pred = model(coords, vels, forces, distances)
                
                # Calculate losses
                pos_loss = F.mse_loss(pos_pred, coords)
                vel_loss = F.mse_loss(vel_pred, vels)
                force_loss = F.mse_loss(force_pred, forces)
                
                # Combined loss
                loss = w_pos * pos_loss + w_vel * vel_loss + w_force * force_loss
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

def bidirectional_extrapolation(model, seed_data, protein_coords, forward_steps=20, backward_steps=20):
    """
    Perform bidirectional extrapolation of ion dynamics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Extract seed data
    seed_coords = seed_data['coords'].to(device)
    seed_vels = seed_data['vels'].to(device)
    seed_forces = seed_data['forces'].to(device)
    seed_distances = seed_data['distances'].to(device)
    
    batch_size, seq_len = seed_coords.shape[0], seed_coords.shape[1]
    
    # Forward extrapolation
    with torch.no_grad():
        # Get model predictions for seed data
        pos_pred, vel_pred, force_pred = model(seed_coords, seed_vels, seed_forces, seed_distances, 
                                               extrapolation_steps=forward_steps)
    
    # Get extrapolated forward data
    forward_coords = pos_pred[:, -forward_steps:]
    forward_vels = vel_pred[:, -forward_steps:]
    forward_forces = force_pred[:, -forward_steps:]
    
    # Backward extrapolation by reversing time
    # Flip seed data in time dimension
    reversed_coords = torch.flip(seed_coords, dims=[1])
    # For backward extrapolation, we need to flip velocity sign
    reversed_vels = -torch.flip(seed_vels, dims=[1])
    reversed_forces = torch.flip(seed_forces, dims=[1])
    reversed_distances = torch.flip(seed_distances, dims=[1])
    
    with torch.no_grad():
        # Get model predictions for reversed seed data
        reversed_pos_pred, reversed_vel_pred, reversed_force_pred = model(
            reversed_coords, reversed_vels, reversed_forces, reversed_distances,
            extrapolation_steps=backward_steps
        )
    
    # Extract backwards extrapolation and reverse it back
    backward_coords = torch.flip(reversed_pos_pred[:, -backward_steps:], dims=[1])
    # Flip velocity sign back
    backward_vels = -torch.flip(reversed_vel_pred[:, -backward_steps:], dims=[1])
    backward_forces = torch.flip(reversed_force_pred[:, -backward_steps:], dims=[1])
    
    # Combine backward, seed, and forward data
    full_coords = torch.cat([backward_coords, seed_coords, forward_coords], dim=1)
    full_vels = torch.cat([backward_vels, seed_vels, forward_vels], dim=1)
    full_forces = torch.cat([backward_forces, seed_forces, forward_forces], dim=1)
    
    return full_coords, full_vels, full_forces

def plot_results(original_coords, original_vels, original_forces,
                predicted_coords, predicted_vels, predicted_forces,
                seed_len, forward_steps, backward_steps):
    """
    Plot the original and predicted ion dynamics.
    """
    # Create time axis for plotting
    total_len = seed_len + forward_steps + backward_steps
    time_orig = np.arange(len(original_coords))
    time_pred = np.arange(-backward_steps, seed_len + forward_steps)
    
    # Plot coordinates
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    components = ['x', 'y', 'z']
    
    # Plot each component of position, velocity, and force
    for i in range(3):
        # Coordinates
        axes[0, i].plot(time_orig, original_coords[:, i], 'k-', label='Original')
        axes[0, i].plot(time_pred, predicted_coords[0, :, i].cpu().numpy(), 'r--', label='Predicted')
        # Mark seed region
        axes[0, i].axvspan(0, seed_len-1, alpha=0.2, color='gray')
        axes[0, i].set_title(f'Position {components[i]}')
        axes[0, i].legend()
        
        # Velocities
        axes[1, i].plot(time_orig, original_vels[:, i], 'k-', label='Original')
        axes[1, i].plot(time_pred, predicted_vels[0, :, i].cpu().numpy(), 'r--', label='Predicted')
        axes[1, i].axvspan(0, seed_len-1, alpha=0.2, color='gray')
        axes[1, i].set_title(f'Velocity {components[i]}')
        
        # Forces
        axes[2, i].plot(time_orig, original_forces[:, i], 'k-', label='Original')
        axes[2, i].plot(time_pred, predicted_forces[0, :, i].cpu().numpy(), 'r--', label='Predicted')
        axes[2, i].axvspan(0, seed_len-1, alpha=0.2, color='gray')
        axes[2, i].set_title(f'Force {components[i]}')
    
    plt.tight_layout()
    plt.savefig('ion_dynamics_prediction.png')
    plt.close()
    
    # Plot energy conservation
    mass = 39.1  # K+ mass in atomic units
    
    # Calculate kinetic energy: 0.5 * m * v^2
    ke_orig = 0.5 * mass * np.sum(original_vels**2, axis=1)
    ke_pred = 0.5 * mass * torch.sum(predicted_vels**2, dim=2).cpu().numpy()[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_orig, ke_orig, 'k-', label='Original KE')
    plt.plot(time_pred, ke_pred, 'r--', label='Predicted KE')
    plt.axvspan(0, seed_len-1, alpha=0.2, color='gray', label='Seed Region')
    plt.title('Kinetic Energy Conservation')
    plt.legend()
    plt.savefig('energy_conservation.png')
    plt.close()

# Main function to run the whole pipeline
def main(ion_coords_file, ion_vels_file, ion_forces_file, protein_coords_file):
    # Load data from CSV files
    ion_coords = pd.read_csv(ion_coords_file, index_col='frame').values.astype(np.float32)
    ion_vels = pd.read_csv(ion_vels_file, index_col='frame').values.astype(np.float32)
    ion_forces = pd.read_csv(ion_forces_file, index_col='frame').values.astype(np.float32)
    
    # Load protein coordinates from CSV
    protein_coords = pd.read_csv(protein_coords_file, index_col='atom_index').values.astype(np.float32)
    # Create dataset
    seq_length = 32  # Length of sequence window
    dataset = IonTransportDataset(ion_coords, ion_vels, ion_forces, protein_coords, seq_length=seq_length)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = PhysicsInformedSchNet(
        hidden_dim=128, 
        n_interaction_blocks=3,
        n_rbf=50,
        cutoff=10.0,
        lstm_layers=2
    )
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50)
    
    # Save model
    torch.save(model.state_dict(), 'physics_informed_schnet.pt')
    
    # Perform bidirectional extrapolation on a validation batch
    val_batch = next(iter(val_loader))
    forward_steps = 20
    backward_steps = 20
    
    full_coords, full_vels, full_forces = bidirectional_extrapolation(
        model, val_batch, protein_coords, 
        forward_steps=forward_steps, 
        backward_steps=backward_steps
    )
    
    # Plot results
    sample_idx = 0  # Choose first sample from batch for visualization
    plot_results(
        ion_coords, ion_vels, ion_forces,
        full_coords, full_vels, full_forces,
        seq_length, forward_steps, backward_steps
    )
    
    print("Training and evaluation completed successfully!")
def generate_ion_trajectory(model, protein_coords, initial_state, n_steps=100, temperature=0.1, dt=0.01):
    """
    Generate a new ion trajectory using the trained model.
    
    Args:
        model: Trained PhysicsInformedSchNet model
        protein_coords: Coordinates of protein atoms (torch.Tensor)
        initial_state: Dictionary with 'coords', 'vels', 'forces' for initial state
        n_steps: Number of steps to generate
        temperature: Temperature factor for noise injection (exploration)
        dt: Time step (should match model's dt)
        
    Returns:
        Generated trajectory as tensors of coordinates, velocities, and forces
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Ensure protein coordinates are on the correct device
    protein_coords = protein_coords.to(device)
    
    # Extract initial state
    pos = initial_state['coords'].to(device)
    vel = initial_state['vels'].to(device)
    force = initial_state['forces'].to(device)
    
    # Initialize arrays for generated trajectory
    trajectory_pos = torch.zeros(1, n_steps, 3, device=device)
    trajectory_vel = torch.zeros(1, n_steps, 3, device=device)
    trajectory_force = torch.zeros(1, n_steps, 3, device=device)
    
    # Set initial values
    trajectory_pos[:, 0] = pos
    trajectory_vel[:, 0] = vel
    trajectory_force[:, 0] = force
    
    # Extract physics parameters from model
    mass = model.physics_decoder.mass.item()
    
    # Generate trajectory step by step
    for i in range(1, n_steps):
        # Calculate distances to protein atoms
        curr_pos = trajectory_pos[:, i-1]
        curr_distances = torch.norm(curr_pos.unsqueeze(1) - protein_coords.unsqueeze(0), dim=2)
        
        # Prepare input features for the model
        curr_vel = trajectory_vel[:, i-1]
        curr_force = trajectory_force[:, i-1]
        
        # Create the sequence of just one frame (current frame)
        seq_pos = curr_pos.unsqueeze(1)  # [1, 1, 3]
        seq_vel = curr_vel.unsqueeze(1)  # [1, 1, 3]
        seq_force = curr_force.unsqueeze(1)  # [1, 1, 3]
        seq_distances = curr_distances.unsqueeze(1)  # [1, 1, n_protein_atoms]
        
        # Forward pass through model
        with torch.no_grad():
            # Get the model's prediction for next state
            next_pos, next_vel, next_force = model(
                seq_pos, seq_vel, seq_force, seq_distances, 
                enforce_physics=True,
                extrapolation_steps=1
            )
        
        # Extract the extrapolated values (index 1 because index 0 is input)
        next_pos = next_pos[:, 1]
        next_vel = next_vel[:, 1]
        next_force = next_force[:, 1]
        
        # Add exploration noise scaled by temperature
        if temperature > 0:
            force_noise = torch.randn_like(next_force) * temperature
            next_force = next_force + force_noise
            
            # Recompute velocity and position based on noisy force
            accel = next_force / mass
            next_vel = curr_vel + accel * dt
            next_pos = curr_pos + next_vel * dt + 0.5 * accel * dt**2
        
        # Store values in trajectory
        trajectory_pos[:, i] = next_pos
        trajectory_vel[:, i] = next_vel
        trajectory_force[:, i] = next_force
    
    return trajectory_pos, trajectory_vel, trajectory_force

def visualize_trajectory(trajectory_pos, protein_coords, save_path='trajectory_visualization.png'):
    """
    Visualize the ion trajectory in the context of the protein.
    
    Args:
        trajectory_pos: Generated position trajectory [1, n_steps, 3]
        protein_coords: Coordinates of protein atoms
        save_path: Path to save the visualization
    """
    # Convert to numpy for matplotlib
    trajectory = trajectory_pos[0].cpu().numpy()
    protein = protein_coords.cpu().numpy()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot protein atoms as small dots
    ax.scatter(protein[:, 0], protein[:, 1], protein[:, 2], c='gray', s=10, alpha=0.5, label='Protein')
    
    # Plot ion trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', lw=2, label='Ion trajectory')
    
    # Highlight start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='green', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='red', s=100, label='End')
    
    # Add labels and legend
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Potassium Ion Trajectory in Gramicidin A Channel')
    ax.legend()
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    
    print(f"Trajectory visualization saved to {save_path}")

def analyze_trajectory(trajectory_pos, trajectory_vel, trajectory_force, mass=39.1, dt=0.01):
    """
    Analyze the generated trajectory for important physical properties.
    
    Args:
        trajectory_pos: Generated position trajectory [1, n_steps, 3]
        trajectory_vel: Generated velocity trajectory [1, n_steps, 3]
        trajectory_force: Generated force trajectory [1, n_steps, 3]
        mass: Mass of potassium ion in atomic units
        dt: Time step
    """
    # Convert to numpy for analysis
    pos = trajectory_pos[0].cpu().numpy()
    vel = trajectory_vel[0].cpu().numpy()
    force = trajectory_force[0].cpu().numpy()
    
    # Calculate kinetic energy
    ke = 0.5 * mass * np.sum(vel**2, axis=1)
    
    # Calculate displacement over time
    displacement = np.sqrt(np.sum((pos - pos[0])**2, axis=1))
    
    # Calculate average velocity magnitude
    velocity_mag = np.sqrt(np.sum(vel**2, axis=1))
    
    # Calculate acceleration
    accel = force / mass
    accel_mag = np.sqrt(np.sum(accel**2, axis=1))
    
    # Create time axis
    time = np.arange(len(pos)) * dt
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Displacement plot
    axes[0, 0].plot(time, displacement)
    axes[0, 0].set_xlabel('Time (ps)')
    axes[0, 0].set_ylabel('Displacement (Å)')
    axes[0, 0].set_title('Ion Displacement from Starting Position')
    
    # Kinetic energy plot
    axes[0, 1].plot(time, ke)
    axes[0, 1].set_xlabel('Time (ps)')
    axes[0, 1].set_ylabel('Energy (a.u.)')
    axes[0, 1].set_title('Kinetic Energy')
    
    # Velocity magnitude plot
    axes[1, 0].plot(time, velocity_mag)
    axes[1, 0].set_xlabel('Time (ps)')
    axes[1, 0].set_ylabel('Velocity (Å/ps)')
    axes[1, 0].set_title('Velocity Magnitude')
    
    # Acceleration magnitude plot
    axes[1, 1].plot(time, accel_mag)
    axes[1, 1].set_xlabel('Time (ps)')
    axes[1, 1].set_ylabel('Acceleration (Å/ps²)')
    axes[1, 1].set_title('Acceleration Magnitude')
    
    plt.tight_layout()
    plt.savefig('trajectory_analysis.png')
    plt.close()
    
    # Calculate and print statistical properties
    print("Trajectory Analysis:")
    print(f"Duration: {time[-1]:.2f} ps")
    print(f"Total displacement: {displacement[-1]:.2f} Å")
    print(f"Average velocity: {np.mean(velocity_mag):.2f} Å/ps")
    print(f"Maximum velocity: {np.max(velocity_mag):.2f} Å/ps")
    print(f"Average kinetic energy: {np.mean(ke):.2f} a.u.")
    
    # Export data to CSV
    data = {
        'time': time,
        'x': pos[:, 0],
        'y': pos[:, 1],
        'z': pos[:, 2],
        'vx': vel[:, 0],
        'vy': vel[:, 1],
        'vz': vel[:, 2],
        'fx': force[:, 0],
        'fy': force[:, 1],
        'fz': force[:, 2],
        'kinetic_energy': ke,
        'displacement': displacement
    }
    df = pd.DataFrame(data)
    df.to_csv('generated_trajectory.csv', index=False)
    print("Trajectory data saved to generated_trajectory.csv")
    
    return df

# Function to run the full trajectory generation pipeline
def generate_and_analyze_ion_trajectory(model_path, protein_coords_file, n_steps=13000, temperature=0.1):
    """
    Run the full trajectory generation pipeline.
    
    Args:
        model_path: Path to the trained model
        protein_coords_file: File containing protein coordinates
        n_steps: Length of trajectory to generate
        temperature: Temperature factor for noise injection
    """
    # Load protein coordinates
    protein_coords = pd.read_csv(protein_coords_file, index_col='atom_index').values.astype(np.float32)
    protein_coords = torch.tensor(protein_coords, dtype=torch.float32)
    
    # Initialize model
    model = PhysicsInformedSchNet(
        hidden_dim=128, 
        n_interaction_blocks=3,
        n_rbf=50,
        cutoff=10.0,
        lstm_layers=2
    )
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define initial state for the ion
    # Starting at the entrance of the channel with reasonable velocity and force
    initial_state = {
        'coords': torch.tensor([[0.0, 0.0, -15.0]], dtype=torch.float32),  # At channel entrance
        'vels': torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32),     # Small velocity along channel axis
        'forces': torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float32)    # Small force along channel axis
    }
    
    print("Generating ion trajectory...")
    traj_pos, traj_vel, traj_force = generate_ion_trajectory(
        model, 
        protein_coords, 
        initial_state, 
        n_steps=n_steps, 
        temperature=temperature,
        dt=model.physics_decoder.dt.item()
    )
    
    print("Visualizing trajectory...")
    visualize_trajectory(traj_pos, protein_coords)
    
    print("Analyzing trajectory...")
    analysis_df = analyze_trajectory(
        traj_pos, 
        traj_vel, 
        traj_force, 
        mass=model.physics_decoder.mass.item(),
        dt=model.physics_decoder.dt.item()
    )
    
    return traj_pos, traj_vel, traj_force, analysis_df

# Extend the main function to include trajectory generation
def main_with_trajectory_generation(ion_coords_file, ion_vels_file, ion_forces_file, protein_coords_file):
    # Run the original training pipeline
    main(ion_coords_file, ion_vels_file, ion_forces_file, protein_coords_file)
    
    # Generate new trajectories using the trained model
    print("\n--- Starting Trajectory Generation ---\n")
    traj_pos, traj_vel, traj_force, analysis = generate_and_analyze_ion_trajectory(
        model_path='physics_informed_schnet.pt',
        protein_coords_file=protein_coords_file,
        n_steps=1200,
        temperature=0.1
    )
    
    # Create animation of the trajectory
    create_trajectory_animation(traj_pos[0].cpu().numpy(), 
                               protein_coords=torch.tensor(pd.read_csv(protein_coords_file, index_col='atom_index').values.astype(np.float32)),
                               output_file='ion_trajectory_animation.gif')
    
    print("Trajectory generation and analysis completed successfully!")
    return traj_pos, traj_vel, traj_force, analysis

def create_trajectory_animation(trajectory, protein_coords, output_file='trajectory_animation.gif', fps=20):
    """
    Create an animated visualization of the ion trajectory.
    
    Args:
        trajectory: Ion positions [n_steps, 3]
        protein_coords: Protein atom coordinates
        output_file: Output filename
        fps: Frames per second
    """
    from matplotlib.animation import FuncAnimation
    
    # Convert protein coords to numpy if needed
    if isinstance(protein_coords, torch.Tensor):
        protein_coords = protein_coords.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot protein atoms (static)
    ax.scatter(protein_coords[:, 0], protein_coords[:, 1], protein_coords[:, 2], 
               c='gray', s=10, alpha=0.3, label='Protein')
    
    # Set axis limits based on trajectory and protein
    all_points = np.vstack([trajectory, protein_coords])
    max_range = np.max([
        np.max(all_points[:, 0]) - np.min(all_points[:, 0]),
        np.max(all_points[:, 1]) - np.min(all_points[:, 1]),
        np.max(all_points[:, 2]) - np.min(all_points[:, 2])
    ])
    mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
    mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
    mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Setup plot elements that will be updated in the animation
    path, = ax.plot([], [], [], 'b-', lw=2)
    ion_point, = ax.plot([], [], [], 'ro', ms=10)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Set labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Potassium Ion Transport through Gramicidin A')
    
    # Animation initialization function
    def init():
        path.set_data([], [])
        path.set_3d_properties([])
        ion_point.set_data([], [])
        ion_point.set_3d_properties([])
        time_text.set_text('')
        return path, ion_point, time_text
    
    # Animation update function
    def update(frame):
        # Update path to show trajectory up to current frame
        path.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
        path.set_3d_properties(trajectory[:frame, 2])
        
        # Update current ion position
        ion_point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        ion_point.set_3d_properties([trajectory[frame, 2]])
        
        # Update time text
        time_text.set_text(f'Time: {frame * 0.01:.2f} ps')
        
        return path, ion_point, time_text
    
    # Create animation
    frames = min(len(trajectory), 200)  # Limit frames to avoid too large files
    stride = len(trajectory) // frames if len(trajectory) > frames else 1
    ani = FuncAnimation(fig, update, frames=range(0, len(trajectory), stride),
                        init_func=init, blit=False, interval=1000/fps)
    
    # Save animation
    ani.save(output_file, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved to {output_file}")

if __name__ == "__main__":
    # Point to CSV files
    ion_coords_file = "pot_ion_coordinates.csv"
    ion_vels_file = "pot_ion_velocities.csv"
    ion_forces_file = "pot_ion_forces.csv"
    protein_coords_file = "protein_ca_coords.csv"
    
    # Run the extended pipeline
    main_with_trajectory_generation(ion_coords_file, ion_vels_file, ion_forces_file, protein_coords_file)