import numpy as np
from sarpy.io.complex.sicd import SICDReader
from skimage.registration import phase_cross_correlation
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import logging
import psutil
import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import interp1d
import torch
from scipy.interpolate import interp2d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sar_micro_motion.log')
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_memory_usage(message):
    """Log current memory usage with a descriptive message."""
    memory_mb = get_memory_usage()
    logger.info(f"{message} - Memory usage: {memory_mb:.2f} MB")

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    return str(timedelta(seconds=int(seconds)))

def process_sub_aperture(t, range_doppler, master, n_az, sub_ap_width, step_size, patch_size, step):
    """Helper function to process a single sub-aperture sequentially."""
    try:
        # Create mask for current sub-aperture with overlap
        mask = np.zeros_like(range_doppler, dtype=complex)
        start = t * step_size
        end = start + sub_ap_width
        if end > n_az:
            start = n_az - sub_ap_width
            end = n_az
        mask[:, start:end] = range_doppler[:, start:end]
        
        # Generate current sub-aperture image
        sub_ap_image = np.fft.ifft(mask, axis=1)
        slave = np.abs(sub_ap_image)
        
        # Process patches
        patches = view_as_windows(master, patch_size, step)
        slave_patches = view_as_windows(slave, patch_size, step)
        n_patches_y, n_patches_x = patches.shape[:2]
        logger.info(f"Processing sub-aperture with {n_patches_y} x {n_patches_x} patches (height x width)")

        shifts = np.zeros((n_patches_y, n_patches_x, 2))
        
        # Process patches sequentially
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                try:
                    master_patch = patches[i, j]
                    slave_patch = slave_patches[i, j]
                    shift, _, _ = phase_cross_correlation(
                        master_patch,
                        slave_patch,
                        upsample_factor=64
                    )
                    shifts[i, j] = shift
                except Exception as e:
                    logger.error(f"Error processing patch ({i}, {j}): {str(e)}")
                    shifts[i, j] = np.zeros(2)
        
        return shifts
    except Exception as e:
        logger.error(f"Error in sub-aperture processing: {str(e)}")
        return None

def extract_sub_aperture(range_doppler, start, sub_ap_width, device=None):
    """Helper function to extract a sub-aperture with bounds checking using PyTorch.
    
    Args:
        range_doppler (np.ndarray or torch.Tensor): Range-Doppler spectrum.
        start (int): Starting index for the sub-aperture.
        sub_ap_width (int): Width of the sub-aperture.
        device (torch.device): Device to use for computation (default: None).
    
    Returns:
        torch.Tensor: Extracted sub-aperture amplitude image.
    """
    # Convert to torch tensor if not already
    if not isinstance(range_doppler, torch.Tensor):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        range_doppler = torch.tensor(range_doppler, device=device, dtype=torch.complex64)
    
    with torch.no_grad():
        n_az = range_doppler.shape[1]
        if start + sub_ap_width > n_az:
            start = n_az - sub_ap_width
        end = start + sub_ap_width
        
        # Create mask for current sub-aperture
        mask = torch.zeros_like(range_doppler)
        mask[:, start:end] = range_doppler[:, start:end]
        
        # Generate sub-aperture image
        sub_ap_image = torch.fft.ifft(mask, dim=1)
        amplitude = torch.abs(sub_ap_image)
        
    return amplitude

def generate_and_process_sub_apertures(sicd_data, M, patch_size=(128, 128), step=64, upsample_factor=200, batch_size=16):
    """
    Generate sub-apertures and compute sub-pixel shifts using a sliding master strategy.
    
    Args:
        sicd_data (np.ndarray): Input SAR data (range x azimuth).
        M (int): Number of sub-apertures.
        patch_size (tuple): Patch size for coregistration.
        step (int): Step size for patch extraction.
        upsample_factor (int): Oversampling factor for sub-pixel accuracy.
        batch_size (int): Number of patches to process in each mini-batch.
    
    Returns:
        np.ndarray: Accumulated sub-pixel shifts over time.
    """
    start_time = time.time()
    logger.info(f"Starting sub-aperture processing with {M} sub-apertures")
    log_memory_usage("Before sub-aperture processing")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Compute range-Doppler transform in PyTorch
    with torch.no_grad():
        sicd_tensor = torch.tensor(sicd_data, device=device, dtype=torch.complex64)
        range_doppler = torch.fft.fft(sicd_tensor, dim=1)
        del sicd_tensor  # Free memory
    
    n_az = sicd_data.shape[1]
    sub_ap_width = n_az // M
    step_size = sub_ap_width // 2  # Overlap for sliding strategy
    
    # Extract first master sub-aperture
    master = extract_sub_aperture(range_doppler, start=0, sub_ap_width=sub_ap_width, device=device)
    master_np = master.cpu().numpy()
    del master  # Free GPU memory
    
    # Initialize shifts array (we'll set dimensions after first processing)
    shifts = None
    
    # Process all sub-aperture pairs
    for t in tqdm(range(M-1), desc="Processing sub-aperture pairs"):
        # Extract slave sub-aperture (which will become the next master)
        slave = extract_sub_aperture(range_doppler, start=(t+1)*step_size, sub_ap_width=sub_ap_width, device=device)
        slave_np = slave.cpu().numpy()
        
        # Compute sub-pixel shifts
        result = process_sub_aperture_mps(master_np, slave_np, patch_size, step, upsample_factor, batch_size=batch_size)
        
        # Initialize shifts array if first iteration
        if shifts is None:
            result_shape = result.shape
            logger.info(f"Result shape: {result_shape}")
            shifts = np.zeros((M-1, *result_shape))
        
        # Store result
        shifts[t] = result
        
        # Reuse slave as master for next iteration
        master_np = slave_np
        del slave  # Free GPU memory
    
    # Free large tensor
    del range_doppler
    
    # Accumulate shifts over time
    total_shifts = np.cumsum(shifts, axis=0)
    total_time = time.time() - start_time
    logger.info(f"Sub-aperture processing completed in {format_time(total_time)}")
    log_memory_usage("After completing all sub-apertures")
    return total_shifts

def analyze_shifts(shifts, slant_res_rg, slant_res_az, collect_duration, M):
    """
    Analyze pixel shifts to estimate physical displacements and vibration frequencies.

    Parameters:
    - shifts (ndarray): Pixel shifts (M, n_patches_y, n_patches_x, 2)
    - slant_res_rg (float): Range resolution in meters per pixel
    - slant_res_az (float): Azimuth resolution in meters per pixel
    - collect_duration (float): Total collection time in seconds
    - M (int): Number of sub-apertures

    Returns:
    - displacements (ndarray): Physical displacements in meters
    - freq_spectra (ndarray): Frequency spectra of displacements
    - dominant_freqs (ndarray): Dominant frequencies per patch
    """
    start_time = time.time()
    logger.info("Starting shift analysis")
    log_memory_usage("Before shift analysis")
    
    # Convert pixel shifts to physical displacements
    logger.info("Converting pixel shifts to physical displacements...")
    displacements = np.zeros_like(shifts)
    displacements[..., 0] = shifts[..., 0] * slant_res_rg  # Range displacement
    displacements[..., 1] = shifts[..., 1] * slant_res_az  # Azimuth displacement
    
    # Define time axis and sampling frequency
    dt = collect_duration / (M - 1)  # Time step between sub-apertures
    time_axis = np.arange(M - 1) * dt  # Now has length M-1
    fs = 1 / dt  # Sampling frequency
    logger.info(f"Sampling frequency: {fs:.2f} Hz")
    
    # Interpolate the displacement time series for better frequency resolution
    logger.info("Interpolating displacement time series...")
    
    # Increase the number of points for better frequency resolution
    M_interp = (M - 1) * 4  # 4x oversampling based on M-1
    time_interp = np.linspace(0, collect_duration, M_interp)
    
    # Initialize interpolated displacements array
    displacements_interp = np.zeros((M_interp, displacements.shape[1], displacements.shape[2], 2))
    
    # Interpolate each patch and direction
    for i in range(displacements.shape[1]):
        for j in range(displacements.shape[2]):
            for d in range(2):  # Range and azimuth directions
                if len(time_axis) != len(displacements[:, i, j, d]):
                    logger.error(f"Time axis length {len(time_axis)} != displacement length {len(displacements[:, i, j, d])}")
                    raise ValueError("Time axis and displacement length mismatch")
                f = interp1d(time_axis, displacements[:, i, j, d], 
                           kind='cubic', bounds_error=False, fill_value='extrapolate')
                displacements_interp[:, i, j, d] = f(time_interp)
    
    # Compute FFT along time axis to get frequency spectra
    logger.info("Computing frequency spectra...")
    freq_spectra = np.fft.fft(displacements_interp, axis=0)
    freq_axis = np.fft.fftfreq(M_interp, time_interp[1] - time_interp[0])
    magnitudes = np.abs(freq_spectra)
    
    # Identify dominant frequency for each patch and direction
    logger.info("Identifying dominant frequencies...")
    dominant_freqs = freq_axis[np.argmax(magnitudes, axis=0)]
    
    total_time = time.time() - start_time
    logger.info(f"Shift analysis completed in {format_time(total_time)}")
    log_memory_usage("After shift analysis")
    return displacements, freq_spectra, dominant_freqs

def visualize_results(displacements, freq_spectra, collect_duration, M, sicd_data):
    """
    Visualize displacement time series, frequency spectra, and oscillation patterns overlaid on SAR image.

    Parameters:
    - displacements (ndarray): Physical displacements in meters
    - freq_spectra (ndarray): Frequency spectra
    - collect_duration (float): Total collection time in seconds
    - M (int): Number of sub-apertures
    - sicd_data (ndarray): Original SAR complex image data
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Time and frequency axes
    dt = collect_duration / (M - 1)
    time = np.arange(M-1) * dt  # Changed from np.arange(M) to np.arange(M-1)
    
    # Get the FFT length from freq_spectra
    M_interp = freq_spectra.shape[0]
    dt_interp = collect_duration / (M_interp - 1)
    freq_axis_interp = np.fft.fftfreq(M_interp, d=dt_interp)
    
    # 1. Plot SAR image with oscillation strength overlay (range direction)
    plt.subplot(2, 2, 1)
    sar_image = np.abs(sicd_data)  # Magnitude of SAR image
    plt.imshow(sar_image, cmap='gray', aspect='auto')
    # Calculate total oscillation strength as sum of frequency magnitudes
    oscillation_strength_rg = np.sum(np.abs(freq_spectra[:, :, :, 0]), axis=0)
    # Resize oscillation strength to match SAR image dimensions
    oscillation_strength_rg_resized = np.repeat(np.repeat(oscillation_strength_rg, 128, axis=0), 128, axis=1)
    # Crop to match SAR image size if necessary
    oscillation_strength_rg_resized = oscillation_strength_rg_resized[:sar_image.shape[0], :sar_image.shape[1]]
    # Create semi-transparent overlay
    plt.imshow(oscillation_strength_rg_resized, cmap='hot', alpha=0.5, aspect='auto')
    plt.colorbar(label='Range Oscillation Strength')
    plt.title('Range Oscillation Pattern Overlay')
    plt.xlabel('Azimuth')
    plt.ylabel('Range')
    
    # 2. Plot SAR image with oscillation strength overlay (azimuth direction)
    plt.subplot(2, 2, 2)
    plt.imshow(sar_image, cmap='gray', aspect='auto')
    oscillation_strength_az = np.sum(np.abs(freq_spectra[:, :, :, 1]), axis=0)
    oscillation_strength_az_resized = np.repeat(np.repeat(oscillation_strength_az, 128, axis=0), 128, axis=1)
    oscillation_strength_az_resized = oscillation_strength_az_resized[:sar_image.shape[0], :sar_image.shape[1]]
    plt.imshow(oscillation_strength_az_resized, cmap='hot', alpha=0.5, aspect='auto')
    plt.colorbar(label='Azimuth Oscillation Strength')
    plt.title('Azimuth Oscillation Pattern Overlay')
    plt.xlabel('Azimuth')
    plt.ylabel('Range')
    
    # 3. Plot displacement time series for center patch
    plt.subplot(2, 2, 3)
    n_patches_y, n_patches_x = displacements.shape[1:3]
    i, j = n_patches_y // 2, n_patches_x // 2
    plt.plot(time, displacements[:, i, j, 0], label='Range')
    plt.plot(time, displacements[:, i, j, 1], label='Azimuth')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Displacement Time Series (Center Patch)')
    plt.legend()
    
    # 4. Plot frequency spectrum for center patch using the correct FFT size
    plt.subplot(2, 2, 4)
    plt.plot(freq_axis_interp[:M_interp//2], np.abs(freq_spectra[:M_interp//2, i, j, 0]), label='Range')
    plt.plot(freq_axis_interp[:M_interp//2], np.abs(freq_spectra[:M_interp//2, i, j, 1]), label='Azimuth')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum (Center Patch)')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('oscillation_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Saved oscillation analysis visualization to 'oscillation_analysis.png'")
    
    # Create a separate figure for dominant frequencies overlay
    plt.figure(figsize=(12, 5))
    
    # Calculate dominant frequencies using the correct frequency axis
    dominant_freqs_rg = freq_axis_interp[np.argmax(np.abs(freq_spectra[:, :, :, 0]), axis=0)]
    dominant_freqs_az = freq_axis_interp[np.argmax(np.abs(freq_spectra[:, :, :, 1]), axis=0)]
    
    # Plot dominant frequency overlays
    plt.subplot(1, 2, 1)
    plt.imshow(sar_image, cmap='gray', aspect='auto')
    dominant_freqs_rg_resized = np.repeat(np.repeat(dominant_freqs_rg, 128, axis=0), 128, axis=1)
    dominant_freqs_rg_resized = dominant_freqs_rg_resized[:sar_image.shape[0], :sar_image.shape[1]]
    plt.imshow(dominant_freqs_rg_resized, cmap='viridis', alpha=0.5, aspect='auto')
    plt.colorbar(label='Range Dominant Frequency (Hz)')
    plt.title('Range Dominant Frequency Overlay')
    plt.xlabel('Azimuth')
    plt.ylabel('Range')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sar_image, cmap='gray', aspect='auto')
    dominant_freqs_az_resized = np.repeat(np.repeat(dominant_freqs_az, 128, axis=0), 128, axis=1)
    dominant_freqs_az_resized = dominant_freqs_az_resized[:sar_image.shape[0], :sar_image.shape[1]]
    plt.imshow(dominant_freqs_az_resized, cmap='viridis', alpha=0.5, aspect='auto')
    plt.colorbar(label='Azimuth Dominant Frequency (Hz)')
    plt.title('Azimuth Dominant Frequency Overlay')
    plt.xlabel('Azimuth')
    plt.ylabel('Range')
    
    plt.tight_layout()
    
    # Save the dominant frequency figure
    plt.savefig('dominant_frequencies.png', dpi=300, bbox_inches='tight')
    logger.info("Saved dominant frequency visualization to 'dominant_frequencies.png'")
    
    plt.close('all')  # Close all figures to free memory

def sub_pixel_shift(corr, shift_int, upsample_factor=200):
    """
    Compute sub-pixel shift using 2D quadratic interpolation around the integer shift.
    
    Args:
        corr (torch.Tensor): Cross-correlation map on MPS device.
        shift_int (tuple): Integer shift (x, y).
        upsample_factor (int): Oversampling factor for sub-pixel precision (default: 200).
    
    Returns:
        torch.Tensor: Sub-pixel shift (x, y).
    """
    device = corr.device
    x, y = shift_int
    
    with torch.no_grad():
        corr_np = corr.cpu().numpy()
        
        # Define fine grid for interpolation
        x_vals = np.linspace(-1, 1, upsample_factor)
        y_vals = np.linspace(-1, 1, upsample_factor)
        
        # Perform 2D quadratic interpolation
        interp_func = interp2d(np.arange(corr.shape[0]), np.arange(corr.shape[1]), corr_np, kind='quadratic')
        corr_interp = interp_func(x_vals + x, y_vals + y)
        
        # Find peak in interpolated map
        max_idx_interp = np.unravel_index(np.argmax(corr_interp), corr_interp.shape)
        sub_pixel_shift = (x + x_vals[max_idx_interp[0]], y + y_vals[max_idx_interp[1]])
        
        return torch.tensor(sub_pixel_shift, device=device)

def process_sub_aperture_mps(master, slave, patch_size=(128, 128), step=64, upsample_factor=200, correlation_threshold=0.8, batch_size=16):
    """
    Coregister master and slave sub-apertures with sub-pixel accuracy using MPS.
    
    Args:
        master (np.ndarray): Master sub-aperture image.
        slave (np.ndarray): Slave sub-aperture image.
        patch_size (tuple): Size of patches (height, width).
        step (int): Step size for patch extraction.
        upsample_factor (int): Oversampling factor for sub-pixel shifts.
        correlation_threshold (float): Minimum correlation value to accept a shift.
        batch_size (int): Number of patches to process in each mini-batch.
    
    Returns:
        np.ndarray: Array of sub-pixel shifts for each patch.
    """
    device = torch.device('mps')
    
    # Extract patches with overlap
    patches_master = view_as_windows(master, patch_size, step)
    patches_slave = view_as_windows(slave, patch_size, step)
    n_patches_y, n_patches_x = patches_master.shape[:2]
    
    # Pre-allocate results array
    shifts = np.zeros((n_patches_y, n_patches_x, 2))
    
    # Convert patches to tensors and move to MPS
    with torch.no_grad():  # Prevent gradient tracking for all operations
        patches_master = torch.tensor(patches_master, device=device, dtype=torch.float32)
        patches_slave = torch.tensor(patches_slave, device=device, dtype=torch.float32)
        
        # Flatten for batch processing
        master_flat = patches_master.reshape(-1, *patch_size)
        slave_flat = patches_slave.reshape(-1, *patch_size)
        
        # Delete original tensors to free memory
        del patches_master, patches_slave
        
        # Get total number of patches
        total_patches = master_flat.shape[0]
        
        # Process in mini-batches
        for batch_start in range(0, total_patches, batch_size):
            # Get current batch
            batch_end = min(batch_start + batch_size, total_patches)
            master_batch = master_flat[batch_start:batch_end]
            slave_batch = slave_flat[batch_start:batch_end]
            
            # Compute FFTs on GPU
            master_fft = torch.fft.fft2(master_batch, dim=(1, 2))
            slave_fft = torch.fft.fft2(slave_batch, dim=(1, 2))
            
            # Cross-correlation via FFT
            cross_corr = master_fft * torch.conj(slave_fft)
            del master_fft, slave_fft  # Free memory
            
            corr_ifft = torch.fft.ifft2(cross_corr, dim=(1, 2))
            del cross_corr  # Free memory
            
            corr_abs = torch.abs(corr_ifft)
            
            # Find integer shifts
            max_idx = torch.argmax(corr_abs.view(batch_end - batch_start, -1), dim=1)
            shifts_int = torch.stack(torch.unravel_index(max_idx, patch_size), dim=1) - torch.tensor(patch_size, device=device) // 2
            
            # Compute sub-pixel shifts for each patch in batch
            for i in range(batch_end - batch_start):
                idx = batch_start + i
                y_idx, x_idx = idx // n_patches_x, idx % n_patches_x
                
                corr = corr_ifft[i]
                corr_peak = corr_abs[i].max()
                
                if corr_peak / corr_abs[i].sum() >= correlation_threshold:  # Apply threshold
                    shift_int = shifts_int[i].cpu().numpy()
                    shift = sub_pixel_shift(corr, shift_int, upsample_factor)
                    shifts[y_idx, x_idx] = shift.cpu().numpy()
                else:
                    shifts[y_idx, x_idx] = [0.0, 0.0]  # Reject low-correlation shifts
            
            # Free batch memory
            del master_batch, slave_batch, corr_ifft, corr_abs
        
        # Free remaining tensors
        del master_flat, slave_flat
    
    return shifts

def save_results(data, filename):
    """Save results to disk as a numpy array."""
    logger.info(f"Saving results to {filename}")
    np.save(filename, data)
    logger.info(f"Results saved successfully")

def load_results(filename):
    """Load results from a numpy array file."""
    logger.info(f"Loading results from {filename}")
    if not os.path.exists(filename):
        logger.error(f"File {filename} not found")
        return None
    try:
        data = np.load(filename)
        logger.info(f"Results loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SAR Micro-Motion Analysis')
    parser.add_argument('--load', type=str, help='Load pre-processed data from file')
    parser.add_argument('--save', type=str, default='sub_aperture_results.npy', 
                        help='Filename to save processed data (default: sub_aperture_results.npy)')
    return parser.parse_args()

def main():
    """Main function to execute the micro-motion estimation workflow."""
    start_time = time.time()
    logger.info("Starting SAR micro-motion analysis")
    log_memory_usage("At start of main")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check for MPS availability
    if not torch.backends.mps.is_available():
        logger.error("MPS (Metal Performance Shaders) is not available on this system")
        sys.exit(1)
    logger.info("MPS is available, using GPU acceleration")
    
    # Step 1: Read the metadata file
    metadata_file = "2023-08-31-01-09-38_UMBRA-04_METADATA.json"
    logger.info(f"Reading metadata file: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract collection duration from timestamps
    start_time_str = metadata['collects'][0]['startAtUTC']
    end_time_str = metadata['collects'][0]['endAtUTC']
    start_time_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
    end_time_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
    collect_duration = (end_time_dt - start_time_dt).total_seconds()
    logger.info(f"Collection duration: {collect_duration:.2f} seconds")
    
    # Extract resolution parameters from metadata
    sicd_metadata = metadata['derivedProducts']['SICD'][0]
    slant_res_rg = sicd_metadata['slantResolution']['rangeMeters']
    slant_res_az = sicd_metadata['slantResolution']['azimuthMeters']
    logger.info(f"Resolution: {slant_res_rg:.2f}m (range) x {slant_res_az:.2f}m (azimuth)")
    
    # Step 2: Read the SICD file
    sicd_file = "2023-08-31-01-09-38_UMBRA-04_SICD.nitf"
    logger.info(f"Reading SICD file: {sicd_file}")
    reader = SICDReader(sicd_file)
    sicd_data = reader[:, :]  # Read entire complex image (range x azimuth)
    log_memory_usage("After reading SICD data")
    
    # Calculate number of sub-apertures needed for 0.06s intervals
    desired_time_interval = 0.06  # seconds
    M = int(collect_duration / desired_time_interval) + 1  # Add 1 to ensure we cover the full duration
    logger.info(f"Using {M} sub-apertures for analysis")
    logger.info(f"Expected time step between sub-apertures: {collect_duration/M:.3f} seconds")
    
    # Step 3: Generate and process sub-aperture images or load from file
    if args.load:
        # Load pre-processed data from file
        shifts = load_results(args.load)
        if shifts is None:
            logger.error("Failed to load results, exiting")
            sys.exit(1)
    else:
        # Generate and process sub-aperture images with MPS acceleration
        patch_size = (128, 128)
        step = 64
        upsample_factor = 200
        shifts = generate_and_process_sub_apertures(
            sicd_data, M, 
            patch_size=patch_size,
            step=step,
            upsample_factor=upsample_factor
        )
        # Save processed data
        save_results(shifts, args.save)
    
    # Step 4: Analyze shifts to estimate displacements and frequencies
    displacements, freq_spectra, dominant_freqs = analyze_shifts(
        shifts, slant_res_rg, slant_res_az, collect_duration, M
    )
    
    # Step 5: Visualize the results
    logger.info("Generating visualizations")
    visualize_results(displacements, freq_spectra, collect_duration, M, sicd_data)
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {format_time(total_time)}")
    log_memory_usage("At end of main")
    logger.info("SAR micro-motion analysis completed successfully")

if __name__ == "__main__":
    main()