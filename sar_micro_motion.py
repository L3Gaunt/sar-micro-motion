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
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from functools import partial

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

def process_patch_parallel(args):
    """Helper function to process a single patch for phase cross-correlation."""
    master_patch, slave_patch = args
    try:
        shift, _, _ = phase_cross_correlation(
            master_patch,
            slave_patch,
            upsample_factor=64  # High upsampling for sub-pixel accuracy
        )
        return shift
    except Exception as e:
        logger.error(f"Error in parallel patch processing: {str(e)}")
        return np.zeros(2)

def process_sub_aperture_sequential(t, range_doppler, master, n_az, sub_ap_width, step_size, patch_size, step):
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

def generate_and_process_sub_apertures(sicd_data, M, patch_size=(128, 128), step=64):
    """
    Generate sub-aperture images and compute shifts with overlapping sub-apertures.
    Uses parallel processing for improved performance.
    """
    start_time = time.time()
    logger.info(f"Starting sub-aperture processing with {M} sub-apertures")
    log_memory_usage("Before sub-aperture processing")
    
    # Compute range-Doppler transform once
    logger.info("Computing range-Doppler transform...")
    range_doppler = np.fft.fft(sicd_data, axis=1)
    n_az = sicd_data.shape[1]
    
    # Calculate sub-aperture width with overlap
    sub_ap_width = n_az // (M // 2)
    overlap = sub_ap_width // 2
    step_size = sub_ap_width - overlap
    
    log_memory_usage("After FFT to range-Doppler domain")

    # Compute master image
    logger.info("Computing master image from first sub-aperture...")
    mask = np.zeros_like(range_doppler, dtype=complex)
    mask[:, 0:sub_ap_width] = range_doppler[:, 0:sub_ap_width]
    sub_ap_image = np.fft.ifft(mask, axis=1)
    master = np.abs(sub_ap_image)
    del mask, sub_ap_image
    log_memory_usage("After computing master image")

    # Initialize shifts array
    patches = view_as_windows(master, patch_size, step)
    n_patches_y, n_patches_x = patches.shape[:2]
    shifts = np.zeros((M, n_patches_y, n_patches_x, 2))
    
    # Process sub-apertures in parallel
    logger.info("Processing sub-apertures in parallel...")
    with Pool(processes=1) as pool:
        # Create partial function with fixed arguments
        process_func = partial(process_sub_aperture_sequential,
                             range_doppler=range_doppler,
                             master=master,
                             n_az=n_az,
                             sub_ap_width=sub_ap_width,
                             step_size=step_size,
                             patch_size=patch_size,
                             step=step)
        
        # Process sub-apertures in parallel
        results = list(tqdm(
            pool.imap(process_func, range(M)),
            total=M,
            desc="Processing sub-apertures"
        ))
    
    # Combine results
    for t, result in enumerate(results):
        if result is not None:
            shifts[t] = result
    
    # Free memory
    del range_doppler
    total_time = time.time() - start_time
    logger.info(f"Sub-aperture processing completed in {format_time(total_time)}")
    log_memory_usage("After completing all sub-apertures")
    return shifts

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
    time_axis = np.arange(M) * dt
    fs = 1 / dt  # Sampling frequency
    logger.info(f"Sampling frequency: {fs:.2f} Hz")
    
    # Interpolate the displacement time series for better frequency resolution
    logger.info("Interpolating displacement time series...")
    
    # Increase the number of points for better frequency resolution
    M_interp = M * 4  # 4x oversampling
    time_interp = np.linspace(0, collect_duration, M_interp)
    
    # Initialize interpolated displacements array
    displacements_interp = np.zeros((M_interp, displacements.shape[1], displacements.shape[2], 2))
    
    # Interpolate each patch and direction
    for i in range(displacements.shape[1]):
        for j in range(displacements.shape[2]):
            for d in range(2):  # Range and azimuth directions
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
    time = np.arange(M) * dt
    freq_axis = np.fft.fftfreq(M, dt)
    
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
    
    # 4. Plot frequency spectrum for center patch
    plt.subplot(2, 2, 4)
    plt.plot(freq_axis[:M//2], np.abs(freq_spectra[:M//2, i, j, 0]), label='Range')
    plt.plot(freq_axis[:M//2], np.abs(freq_spectra[:M//2, i, j, 1]), label='Azimuth')
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
    
    # Calculate dominant frequencies
    dominant_freqs_rg = freq_axis[np.argmax(np.abs(freq_spectra[:, :, :, 0]), axis=0)]
    dominant_freqs_az = freq_axis[np.argmax(np.abs(freq_spectra[:, :, :, 1]), axis=0)]
    
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

def main():
    """Main function to execute the micro-motion estimation workflow."""
    start_time = time.time()
    logger.info("Starting SAR micro-motion analysis")
    log_memory_usage("At start of main")
    
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
    
    # Set number of sub-apertures to 800 for high-frequency detection
    # This will give us a time step of ~0.06 seconds with overlap
    M = 800
    logger.info(f"Using {M} sub-apertures for analysis")
    logger.info(f"Expected time step between sub-apertures: {collect_duration/M:.3f} seconds")
    
    # Step 3: Generate and process sub-aperture images with memory-efficient approach
    shifts = generate_and_process_sub_apertures(sicd_data, M)
    
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