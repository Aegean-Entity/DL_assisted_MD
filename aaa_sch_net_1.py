import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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
        self.dt = nn.Parameter(torch.tensor(0.01), requires_grad=True)  # Time step
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
    # Load data
    ion_coords = np.load(ion_coords_file)
    ion_vels = np.load(ion_vels_file)
    ion_forces = np.load(ion_forces_file)
    protein_coords = np.load(protein_coords_file)
    
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

if __name__ == "__main__":
    # Replace with your actual file paths
    ion_coords_file = "pot_ion_coordinates.npy"
    ion_vels_file = "pot_ion_velocities.npy"
    ion_forces_file = "pot_ion_forces.npy"
    protein_coords_file = "protein_ca_coords.npy"  # You need to extract these from the PSF file
    
    main(ion_coords_file, ion_vels_file, ion_forces_file, protein_coords_file)