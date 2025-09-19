# classify_batch.py
#
# FOR THE ADVANCED TASK
# This script processes all images in a specified folder, runs inference on each,
# and saves the results to a CSV file.
#
# Usage:
# python classify_folder.py <path_to_image_folder>
# classify_batch.py
#
# FOR THE ADVANCED TASK
# This script processes all images in a specified folder, runs inference on each,
# and saves the results to a CSV file.
#
# Usage:
# python classify_batch.py <path_to_image_folder>

import os
import sys
import csv
import classify  # 导入classify.py作为模块

# --- Configuration ---
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
output_path=os.path.expanduser("~/lab03/ISDN3000C_Lab03_template")

def get_image_files(folder_path):
    """Returns list of supported image files in the specified folder."""
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            image_files.append(os.path.join(folder_path, filename))
    return sorted(image_files)


def main():
    if len(sys.argv) != 2:
        print("Usage: python classify_batch.py <path_to_image_folder>")
        sys.exit(1)

    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory")
        sys.exit(1)

    # 从classify模块加载模型和标签
    try:
        model = classify.get_model()
        labels = classify.get_labels()
    except Exception as e:
        print(f"Error initializing resources: {e}")
        sys.exit(1)

    # 获取所有图像文件
    image_files = get_image_files(image_folder)
    if not image_files:
        print(f"No supported image files found in {image_folder}")
        sys.exit(0)

    print(f"Found {len(image_files)} image files to process")

    # 准备CSV输出
    output_csv = os.path.join(output_path, "results.csv")
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "predicted_category", "confidence(%)"])

            # 处理每个图像
            for img_path in image_files:
                try:
                    print(f"Processing {os.path.basename(img_path)}...")
                    # 使用classify模块的图像处理和预测函数
                    img_tensor = classify.process_image(img_path)
                    category, confidence = classify.predict(model, img_tensor, labels)
                    
                    # 写入CSV结果
                    writer.writerow([
                        os.path.basename(output_path),
                        category,
                        f"{confidence:.2f}"
                    ])
                    print(f"Completed {os.path.basename(img_path)}")

                except Exception as e:
                    print(f"Error processing {os.path.basename(img_path)}: {e}")
                    writer.writerow([os.path.basename(output_path), "error", "N/A"])

        print(f"Results saved to {output_csv}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
