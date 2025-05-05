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

def find_files_by_days_and_id(sample_files, days, plant_id):
    """
    Find files matching a specific day and plant ID from the given file list,
    and return the corresponding date of these files.
    Note: Assumes all matching files share the same date, so only a single date is returned.

    :param sample_files: List of filenames.
    :param days: Target day string.
    :param plant_id: Target plant ID.
    :return: List of matching files and a single shared date.
    """
    pattern = re.compile(fr'^{days}_([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})_(?:BC|TPA|TRF)-ID{plant_id}_IMG_\d+\.jp[e]?g$',
                         re.IGNORECASE)

    matching_files = []
    date = ""
    for file in sample_files:
        match = pattern.match(file)
        if match:
            matching_files.append(file)
            date = match.group(1)   # Assume all matched files have the same date; take the first match

    return matching_files, date


def plant_id_extract(input_dir):
    # Regular expression to match plant IDs
    id_pattern = re.compile(r"(?:TPA|BC|TRF)-ID(\d+)")

    # Extract IDs from filenames
    ids = set()
    file_names = os.listdir(input_dir)

    # 遍历文件名，提取ID
    for file_name in file_names:
        match = id_pattern.search(file_name)
        if match:
            ids.add(match.group(1))

    id_list = list(ids)

    return id_list


def standard_vector_generation(image_folder_dir, weather_folder_dir, model_dir, model_name, save_dir, required_days):
    input_dir = image_folder_dir
    excel_dir = weather_folder_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stand_save = save_dir

    if not os.path.exists(stand_save):
        os.makedirs(stand_save)
    # print(f"Using device: {device}")

    require_date = required_days

    plant_id_list = plant_id_extract(input_dir)
    real_plant_id_list = []
    temp_output_dir = "temp_outputs"  # Change to a valid path if needed
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    for plant_id in plant_id_list:
        img_list, img_date = find_files_by_days_and_id(os.listdir(input_dir), require_date, plant_id)

        if img_list:
            real_plant_id_list.append(plant_id)
            # weather
            weather_excel = img_date+'.xlsx'
            excel_path = os.path.join(excel_dir,weather_excel)

            weather_df = pd.read_excel(excel_path)
            data = weather_df.select_dtypes(include=['number'])
            data_array = data.values
            weather_tensor = torch.tensor(data_array, dtype=torch.float)

            # model
            model_ft = FeatureExtractNetwork(model_name=str(model_name))

            model_weights = torch.load(model_dir)

            model_ft.load_state_dict(model_weights)
            model_ft.to(device)
            model_ft.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            num_images = len(img_list)

            for img_index, img in enumerate(img_list):
                torch.cuda.empty_cache()
                img_path = os.path.join(input_dir, img)
                image = Image.open(img_path)
                image_transformed = transform(image)
                img1 = image_transformed.unsqueeze(0).to(device)

                weather1 = weather_tensor.unsqueeze(0).to(device)

                output = model_ft((img1, weather1))


                temp_output_path = os.path.join(temp_output_dir, f"output_{plant_id}_{img_index}.pth")
                torch.save(output.cpu(), temp_output_path)

    real_plant_id_list = list(set(real_plant_id_list))

    for plant_id in real_plant_id_list:

        accumulated_outputs = torch.zeros(256, device=device)   #Initialize accumulator tensor with zeros

        output_files = [f for f in os.listdir(temp_output_dir) if f.startswith(f"output_{plant_id}_")]
        for file_name in output_files:
            output_path = os.path.join(temp_output_dir, file_name)
            output = torch.load(output_path, map_location=device)

            accumulated_outputs += output.squeeze()

        average_output = accumulated_outputs / len(output_files)
        # print(f"Average Output for ID {plant_id}: {average_output}")

        save_path = os.path.join(stand_save, f"{plant_id}.pth")

        torch.save(average_output.cpu(), save_path)

    shutil.rmtree(temp_output_dir)

    print('Done !')

