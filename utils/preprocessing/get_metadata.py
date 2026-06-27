import os
import SimpleITK as sitk
import pandas as pd

def get_metadata(input: str, suffix: str = ".nii.gz"):
    """
    Extract metadata from NIfTI files. If input path is a directory, filters all
    files with the given suffix and saves metadata to an xlsx file.

    Args:
        input (str): Input file or directory path
        suffix (str): File suffix to filter, default ".nii.gz"

    Returns:
        None (saves to xlsx for directory input)
        dict (metadata for single file input)
    """
    # Check if input is directory or file
    if os.path.isdir(input):
        # If directory, iterate all files with the given suffix
        file_list = [f for f in os.listdir(input) if f.endswith(suffix)]
        if not file_list:
            print(f"No files with suffix '{suffix}' found in directory {input}")
            return

        # Collect metadata for all files
        metadata = []
        for file in file_list:
            file_path = os.path.join(input, file)
            try:
                img = sitk.ReadImage(file_path)
                metadata.append({
                    "filename": file,
                    "size": img.GetSize(),
                    "origin": img.GetOrigin(),
                    "spacing": img.GetSpacing(),
                    "direction": img.GetDirection()
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Save metadata to xlsx file
        metadata_df = pd.DataFrame(metadata)
        output_file = os.path.join(input, "metadata.xlsx")
        metadata_df.to_excel(output_file, index=False)
        print(f"Metadata saved to {output_file}")

    elif os.path.isfile(input):
        # If single file, return metadata directly
        try:
            img = sitk.ReadImage(input)
            metadata = {
                "filename": os.path.basename(input),
                "size": img.GetSize(),
                "origin": img.GetOrigin(),
                "spacing": img.GetSpacing(),
                "direction": img.GetDirection()
            }
            return metadata
        except Exception as e:
            print(f"Error reading {input}: {e}")
    else:
        print(f"{input} is not a valid file or directory.")
