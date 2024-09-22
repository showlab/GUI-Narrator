import torch
from PIL import Image
import open_clip
from base_model import KeyFrameExtractor_v4, Cursor_detector, ImageReader
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import argparse

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
image_transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def main():
    parser = argparse.ArgumentParser(description='Process paths')
    parser.add_argument('--frame_extract_model_path', type=str, help='Path to the frame extraction model')
    parser.add_argument('--yolo_model_path', type=str, help='Path to the YOLO model')
    parser.add_argument('--images_path', type=str, help='Path to the images')
    args = parser.parse_args()
    
    
    frame_extract_model_path = args.frame_extract_model_path
    yolo_model_path = args.yolo_model_path
    images_path = args.images_path
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('>>>>>>',device)

    model = KeyFrameExtractor_v4()
    loaded_dict = torch.load(frame_extract_model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in loaded_dict.items()})
    model= model.eval()

    try:
        detector = Cursor_detector(yolo_model_path,images_path)
        print('detector_load_successful')
        
    except:
        print('error in loading check_point')
        
        
    detector.detect()
    
    
    image_reader = ImageReader(images_path,transform=image_transform)
    images_tensor = image_reader.read_images()
    output = model(images_tensor.unsqueeze(0))
    values, indices = torch.topk(output, 2)

    start, end = indices[0]
    s= min(int(start), int(end))
    e= max(int(start), int(end))
    return s,e 

if __name__ == "__main__":
    s, e = main()
    print(f'start_frame_index: {s}', f'end_frame_index {e}')
    