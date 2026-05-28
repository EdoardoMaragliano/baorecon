"""
Example: BAO Reconstruction using iFFT Algorithm

This example demonstrates how to use the BAOReconstructor class
to perform Baryon Acoustic Oscillation reconstruction on galaxy survey data.

The example shows:
1. Creating synthetic survey data
2. Initializing the reconstruction
3. Running the full pipeline
4. Analyzing the results
"""
import sys
sys.path.insert(0,'/home/emaragliano/Work/Projects/Dottorato/baorecon/')

import numpy as np
import matplotlib.pyplot as plt
from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor
from zeldareco.utils.loggers import setup_logger

# Setup logging
logger = setup_logger(__name__)


def create_synthetic_survey(n_data=5000, n_random=25000, boxsize=1000.0, 
                            clustering_strength=0.5, seed=42):
    """
    Create synthetic survey data with clustering.
    
    Parameters
    ----------
    n_data : int
        Number of data galaxies
    n_random : int
        Number of random positions
    boxsize : float
        Size of the simulation box in Mpc/h
    clustering_strength : float
        Strength of clustering (0 = uniform, 1 = strong)
    seed : int
        Random seed
    
    Returns
    -------
    data_pos : ndarray
        Galaxy positions, shape (n_data, 3)
    random_pos : ndarray
        Random positions, shape (n_random, 3)
    data_weights : ndarray
        Galaxy weights
    random_weights : ndarray
        Random weights
    """
    np.random.seed(seed)
    
    # Create random catalog (uniform distribution)
    random_pos = np.random.uniform(0, boxsize, size=(n_random, 3)).astype(np.float32)
    random_weights = np.ones(n_random, dtype=np.float32)
    
    # Create data catalog with clustering
    # Start with random positions and add clustering signal
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3)).astype(np.float32)
    
    # Add clustering: concentrate galaxies near certain regions
    n_clusters = 10
    cluster_centers = np.random.uniform(0, boxsize, size=(n_clusters, 3))
    cluster_radius = 50.0  # Mpc/h
    
    for i in range(n_data):
        # Random chance to be near a cluster
        if np.random.rand() < clustering_strength:
            # Pick a random cluster center
            center = cluster_centers[np.random.randint(n_clusters)]
            # Add Gaussian perturbation
            data_pos[i] = center + np.random.normal(0, cluster_radius/3, 3)
            # Wrap around periodic boundaries
            data_pos[i] = np.mod(data_pos[i], boxsize)
    
    data_weights = np.ones(n_data, dtype=np.float32)
    
    return data_pos, random_pos, data_weights, random_weights



def example_basic_reconstruction(solver: str = "ifft"):
    """
    Example 1: Basic reconstruction with default parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic " + solver.upper() + "   BAO Reconstruction")
    print("="*70)
    
    # Create synthetic data
    print("\n1. Creating synthetic survey data...")
    data_pos, random_pos, data_weights, random_weights = create_synthetic_survey(
        n_data=2000,
        n_random=10000,
        boxsize=800.0,
        clustering_strength=0.3,
        seed=42
    )
    print(f"   Created {len(data_pos)} data galaxies and {len(random_pos)} randoms")
    
    
    # Create reconstructor
    print("\n3. Creating reconstructor...")
    recon = BAOReconstructor(
        data_pos=data_pos,
        random_pos=random_pos,
        data_weights=data_weights,
        random_weights=random_weights,
        nmesh=128,
        rectype = "rec-sym",
        threshold_randoms = 0.01,
        solver_type = solver,
        R_sm=15,  # 15 Mpc/h smoothing
        f=0.88,
        bias=1.0,
    )
    recon.print_info()
    
    # Perform reconstruction
    print("\n4. Performing reconstruction...")
    data_rec, random_rec = recon.run_reconstruction()
    
    # Compute displacement statistics
    print("\n6. Displacement statistics...")
    displacement = data_rec - data_pos
    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    print(f"   Mean displacement: {np.mean(displacement_magnitude):.4f} Mpc/h")
    print(f"   Max displacement: {np.max(displacement_magnitude):.4f} Mpc/h")
    print(f"   Displacement std: {np.std(displacement_magnitude):.4f} Mpc/h")
    
    print("\n✅ Example 1 completed successfully!")
    
    return data_pos, data_rec, recon






if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAO RECONSTRUCTION WITH iFFT ALGORITHM - EXAMPLES")
    print("="*70)
    
    # Run examples
    try:
        # Example 1: Basic reconstruction ifft
        data_pos_ifft, data_rec_ifft, recon_ifft = example_basic_reconstruction(solver="ifft")
        
        # Example 2: Parameter comparison
        data_pos_multigrid, data_rec_multigrid, recon_multigrid = example_basic_reconstruction(solver="multigrid")

        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
