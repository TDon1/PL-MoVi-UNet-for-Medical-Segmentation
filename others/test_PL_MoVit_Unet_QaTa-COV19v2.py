import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    classes = 1

    weights_path = "save_weights/PL-MoVi-Unet/best_model_plmovitunet_L_cov19-v2.pth"
    test_images_dir = "DatasetCOV19-v2/test/images"
    output_dir = "testCOV19-v2/predict_results_plmovitunet_L"
    # === 可配置选项 ===
    restore_original_size = False  # True: 还原到原始尺寸, False: 保持模型输出尺寸(224x224)
    # ==================

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_images_dir), f"images directory {test_images_dir} not found."

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print(f"Restore original size: {restore_original_size}")


    model=PLMoViTUnet_large(num_classes=2)
    # 加载权重
    print("Loading weights...")
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 获取测试目录下所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    for file in os.listdir(test_images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    print(f"Found {len(image_files)} images to process")

    input_size = 256
    data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    total_time = 0
    processed_count = 0

    for image_file in image_files:
        img_path = os.path.join(test_images_dir, image_file)

        try:
            # 1. 加载图片并记录原始尺寸
            original_img = Image.open(img_path).convert('RGB')
            org_w, org_h = original_img.size

            # 2. 图像预处理
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            # 3. 模型预测
            t_start = time_synchronized()
            with torch.no_grad():
                output = model(img.to(device))
            t_end = time_synchronized()

            inference_time = t_end - t_start
            total_time += inference_time
            processed_count += 1

            # 4. 处理预测结果
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # 处理像素值 (1 -> 255)
            prediction[prediction == 1] = 255

            # 5. 根据选项决定是否还原尺寸
            if restore_original_size:
                # 还原到原始尺寸
                mask = Image.fromarray(prediction)
                mask = mask.resize((org_w, org_h), resample=Image.NEAREST)
                output_size_info = f"{org_w}x{org_h}"
            else:
                # 保持模型输出尺寸
                mask = Image.fromarray(prediction)
                output_size_info = f"{input_size}x{input_size}"

            # 6. 保存结果
            output_filename = os.path.splitext(image_file)[0] + '.png'
            save_path = os.path.join(output_dir, output_filename)
            mask.save(save_path)

            print(f"Processed {image_file} - Time: {inference_time:.3f}s - Size: {output_size_info}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    # 打印统计信息
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\nProcessing completed!")
        print(f"Total images processed: {processed_count}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per image: {avg_time:.3f}s")
        print(f"Results saved to: {output_dir}")
        print(f"Output size: {'Original size' if restore_original_size else f'{input_size}x{input_size}'}")
    else:
        print("No images were successfully processed.")


if __name__ == '__main__':
    main()
