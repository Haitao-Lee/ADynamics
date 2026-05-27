import os
import SimpleITK as sitk
import pandas as pd

def get_metadata(input: str, suffix: str = ".nii.gz"):
    """
    获取NIfTI文件的元数据信息，如果输入路径是文件夹，筛选所有符合后缀的文件，并将信息保存为xlsx文件。

    参数:
    input (str): 输入文件或文件夹路径
    suffix (str): 要筛选的文件后缀，默认为 ".nii.gz"
    
    返回:
    None
    """
    # 判断输入是文件夹还是文件
    if os.path.isdir(input):
        # 如果是文件夹，遍历文件夹中所有符合后缀的文件
        file_list = [f for f in os.listdir(input) if f.endswith(suffix)]
        if not file_list:
            print(f"No files with suffix '{suffix}' found in directory {input}")
            return
        
        # 收集所有文件的元数据
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

        # 将元数据存储到xlsx文件
        metadata_df = pd.DataFrame(metadata)
        output_file = os.path.join(input, "metadata.xlsx")
        metadata_df.to_excel(output_file, index=False)
        print(f"Metadata saved to {output_file}")
    
    elif os.path.isfile(input):
        # 如果是文件，直接返回文件的元数据信息
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
