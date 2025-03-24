# SAR Micro-Motion Estimation Project

## Overview
This project implements algorithms for estimating micro-motion parameters from Synthetic Aperture Radar (SAR) data. It processes CPHD data to extract motion signatures.

## Build/Run Commands
- Run with sample data: `python sar_micro_motion.py --simulate --output results`
- Run with real data: `python sar_micro_motion.py --input <your-file.npy> --output <name>`
- Run tests: `pytest`
- Code quality check: `flake8`

## Processing CPHD Data
To process a CPHD file, convert it to NumPy format first using the helper scripts in `src/sar/`.

## Code Style Guidelines

### Imports
- Group imports: standard library, third-party, local
- Follow order: numpy, matplotlib, scipy, then others alphabetically

### Formatting
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use whitespace around operators for readability

### Types
- Document types in docstrings
- Numpy arrays are the primary data structure

### Naming Conventions
- Functions/Variables: snake_case
- Constants: UPPER_CASE
- Use descriptive names indicating purpose

### Error Handling
- Use try/except blocks for file operations and data loading
- Validate inputs before processing
- Check file existence before operations