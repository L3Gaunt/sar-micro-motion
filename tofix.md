# Key Discrepancies and Potential Issues

## Todo List

- [ ] **Fix Number of Sub-apertures and Temporal Resolution**
  - [ ] Set M = 200 (or higher) to match the paper's temporal resolution
  - [ ] Adjust sub_ap_width accordingly

- [ ] **Implement Sub-aperture Overlap**
  - [ ] Modify code to generate overlapping sub-apertures
  - [ ] Use step size < sub_ap_width
  - [ ] Increase M to achieve ~0.06-second time step

- [ ] **Improve Frequency Detection Capability**
  - [ ] Increase M significantly with overlap (M > 800 for detecting 33 Hz)
  - [ ] Investigate if paper uses intra-sub-aperture phase analysis <- no, it doesn't

- [ ] **Add Interpolation**
  - [ ] Implement interpolation (e.g., scipy.interpolate) before FFT in analyze_shifts
  - [ ] Refine spectral visualization

- [ ] **Optimize Computational Performance**
  - [ ] Consider implementing parallelization (similar to SARPROZ)
  - [ ] Explore using multiprocessing for handling large datasets