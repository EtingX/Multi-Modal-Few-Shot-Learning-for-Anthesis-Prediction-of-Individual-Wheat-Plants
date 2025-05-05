import os
from tqdm import tqdm
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
import shutil
import re
from model_structure import *


def vector_generation(image_folder_dir, weather_folder_dir, model_dir, model_name, save_dir):
    input_dir = image_folder_dir
    excel_dir = weather_folder_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stand_save = save_dir

    # model
    model_ft = FeatureExtractNetwork(model_name=str(model_name))

    model_weights = torch.load(model_dir)

    # Load the model
    model_ft.load_state_dict(model_weights)
    model_ft.to(device)
    # Ensure the model is in evaluation mode, which is necessary for validation or inference
    model_ft.eval()

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if not os.path.exists(stand_save):
        os.makedirs(stand_save)
    print(f"Using device: {device}")

    pbar = tqdm(enumerate(os.listdir(image_folder_dir)), total=len(os.listdir(image_folder_dir)), desc='Processing Images: ')
    for batch_idx, img in pbar:
        match = re.match(r"(\d+)_((\d{4})-(\d{2})-(\d{2}))", img)

        if match:
            flowering_days = match.group(1)  # Extract the first numeric part
            img_date = match.group(2)   # Extract the date part
            # print(f"Days: {flowering_days}, Date: {img_date}")  # 输出：Days: 14, Date: 2023-07-17
        else:
            print("No match found")


        weather_excel = img_date+'.xlsx'
        excel_path = os.path.join(excel_dir,weather_excel)

        weather_df = pd.read_excel(excel_path)
        data = weather_df.select_dtypes(include=['number'])
        data_array = data.values
        weather_tensor = torch.tensor(data_array, dtype=torch.float)
        img_path = os.path.join(input_dir, img)
        image = Image.open(img_path)
        image_transformed = transform(image)
        img1 = image_transformed.unsqueeze(0).to(device)

        weather1 = weather_tensor.unsqueeze(0).to(device)

        # Get the model output
        output = model_ft((img1, weather1))

        # Save the output to a temporary file
        temp_output_path = os.path.join(stand_save, f"{os.path.splitext(img)[0]}.pth")
        torch.save(output.cpu(), temp_output_path)

        pbar.set_postfix({'img name': str(img), 'day': str(flowering_days),'date':img_date})


    print('Done !')







