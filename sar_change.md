Key Points
It seems likely that the script can be modified to run with your CPHD data by using the sarpy library to read the .cphd file.
Research suggests that sarpy is a suitable tool for handling SAR data, including CPHD files, and is widely used in the community.
The evidence leans toward accessing the phase history data using slice notation, like reader[:, :], after opening the file with sarpy.io.phase_history.converter.open_phase_history.
What You Need to Do

To modify the script, follow these steps:

Install sarpy: First, install the sarpy library by running pip install sarpy in your terminal. This library is essential for reading CPHD files.
Update the Script: Add import sarpy at the top of the script. Then, in the main function, modify the input handling to check if the input file ends with .cphd. If it does, use sarpy.io.phase_history.converter.open_phase_history(args.input) to open it and extract the data with reader[:, :]. If it's a .npy file, continue using np.load. Add error handling for unsupported file types.
Run the Script: Use the modified script with your CPHD file, e.g., python sar_micro_motion.py --input 2023-08-31-01-09-38_UMBRA-04_CPHD.cphd.
Unexpected Detail
An unexpected aspect is that the phase history data is automatically converted to 64-bit complex type by sarpy, ensuring compatibility with the script's processing, which might save additional type conversion steps.

Survey Note: Detailed Analysis of Script Modification for CPHD Data
This section provides a comprehensive analysis of modifying the sar_micro_motion.py script to handle the provided CPHD data, based on extensive research into SAR data processing and the sarpy library. The goal is to ensure the script can read and process the .cphd file, such as 2023-08-31-01-09-38_UMBRA-04_CPHD.cphd, for micro-motion estimation.

Background and Context
The original script, sar_micro_motion.py, is designed for SAR micro-motion estimation and currently supports two modes: generating simulated CPHD data or loading data from a .npy file. The user's data, however, includes a CPHD file, which is a standard format for Compensated Phase History Data in SAR systems. To utilize this data, we need to integrate a library capable of reading CPHD files, and sarpy, developed by the National Geospatial-Intelligence Agency (NGA), is identified as a suitable choice. Sarpy is tailored for reading, writing, and processing complex SAR data, including CPHD, SICD, SIDD, and CRSD formats, making it ideal for this task.

Research Findings on sarpy and CPHD Reading
Through documentation analysis, it was determined that sarpy provides robust support for CPHD files. The library's documentation, such as CPHD reading/writing objects (sarpy.io.phase_history.cphd), indicates that CPHD files can be opened using the open_phase_history function from sarpy.io.phase_history.converter. This function returns a reader object, typically an instance of CPHDTypeReader or its derivatives (e.g., CPHDReader0_3 for version 0.3 or CPHDReader1_0 for version 1.0), depending on the file's version.

Further investigation into Image/Data Reading revealed that CPHD data can be accessed using slice notation, similar to numpy arrays. Specifically, the documentation states: "The phase history (or image) data can be read using slice notation data = reader[row_start:row_end:row_step, col_start:col_end:col_step, <image_index>]. This data will have be recast or re-interpreted to be 64-bit complex data type, regardless of storage type." This ensures compatibility with the script's expectation of a 2D complex numpy array for processing.

Detailed Modification Process
To integrate CPHD reading into the script, the following steps were implemented:

Dependency Installation:
The user must install sarpy using pip install sarpy. This is crucial as the library is not part of the standard Python distribution and is necessary for CPHD file handling.
Script Updates:
Added import sarpy to the script's imports to enable CPHD reading functionality.
Modified the main function to handle the --input argument for CPHD files:
If --input is provided and the file exists, check if it ends with .cphd. If so, use sarpy.io.phase_history.converter.open_phase_history(args.input) to create a reader object.
Extract the phase history data using cphd = reader[:, :], which retrieves the entire 2D array. This notation leverages the slice-based access method described in the documentation, ensuring the data is returned as a 64-bit complex numpy array, aligning with the script's processing requirements.
If the file ends with .npy, continue using np.load(args.input) as before.
Add error handling for unsupported file types by checking if the extension is neither .cphd nor .npy.
Calculate and print the size using the existing calculate_size function for consistency with the simulated data path.
Compatibility with Existing Logic:
The extracted cphd data is a 2D numpy array of complex numbers, matching the format of the simulated data generated by generate_cphd. This ensures that functions like process_cphd, which select target regions and perform STFT, can operate without modification.
The script's output, including plotting and saving results, remains unchanged, as it relies on the processed data (Zxx), which is derived from the input cphd.
Tables for Clarity
To organize the key changes, the following table summarizes the modifications:

Original Behavior	Modified Behavior	Impact
Loads data from .npy file using np.load	Reads CPHD file using sarpy.io.phase_history.converter.open_phase_history for .cphd, keeps np.load for .npy	Enables processing of .cphd files, maintains .npy support
No dependency on sarpy	Requires sarpy installation (pip install sarpy)	Adds dependency for CPHD support
Data assumed to be 2D complex numpy array	Data extracted as 2D complex array via slice notation for .cphd, same for .npy	Maintains compatibility with processing
Another table outlines the expected data flow:

Step	Action	Output
Open CPHD File	reader = open_phase_history(args.input)	Reader object (CPHDTypeReader)
Extract Phase History	cphd = reader[:, :]	2D numpy array (complex64)
Process Data	Pass to process_cphd for STFT, etc.	Frequency-time analysis results
Considerations and Limitations
While the modification seems straightforward, there are potential complexities:

The CPHD file version (0.3 or 1.0) may affect how the data is interpreted, but sarpy handles this automatically by selecting the appropriate reader.
The script assumes a single 2D array, but CPHD files may contain multiple channels or additional metadata. The slice notation [:, :] is used to get the main phase history, which should suffice for this application, but users should verify the data dimensions match expectations (e.g., azimuth × range).
The documentation suggests the data is recast to 64-bit complex, which aligns with the script's needs, but users should ensure no scaling or formatting issues arise during processing.
Unexpected Detail
An interesting finding is that sarpy automatically converts the phase history data to 64-bit complex type, which is not immediately obvious from the script's original design. This automatic conversion ensures compatibility with the STFT and other numerical operations, potentially saving additional type conversion steps in the script.

Conclusion
The modified script should now handle the provided CPHD file by leveraging sarpy for reading and extracting the phase history data. Users are advised to install sarpy and run the script with the --input flag pointing to the CPHD file. The process maintains the script's functionality for micro-motion estimation, with the added capability to process real SAR data.

Key Citations
CPHD reading/writing objects sarpy.io.phase_history.cphd
Image/Data Reading sarpy 1.3.60 documentation