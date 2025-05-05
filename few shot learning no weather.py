from few_shot_learning_testing_method import *
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
import re
from model_structure import *
import random
import shutil

def find_files_by_days_and_id(sample_files, days, plant_id):

    pattern = re.compile(fr'^{days}_([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})_(?:BC|TPA|TRF)-ID{plant_id}_IMG_\d+\.jp[e]?g$',
                         re.IGNORECASE)

    matching_files = []
    date = ""
    for file in sample_files:
        match = pattern.match(file)
        if match:
            matching_files.append(file)
            date = match.group(1)

    return matching_files, date


def plant_id_extract(input_dir):
    id_pattern = re.compile(r"(?:TPA|BC|TRF)-ID(\d+)")

    ids = set()
    file_names = os.listdir(input_dir)

    for file_name in file_names:
        match = id_pattern.search(file_name)
        if match:
            ids.add(match.group(1))

    id_list = list(ids)

    return id_list

class FeatureExtractNetwork_no_weather(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Image network - Choose one based on requirement. Uncomment the needed line.

        if 'swin_b' in model_name:
            self.img_net = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        elif 'convnext' in model_name:
            self.img_net = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            print('Please provide correct model name, efficientnet or swin')

        self.fc_img_1 = nn.Linear(1000, 512)

        self.fc2 = nn.Linear(512, 256)  # Further processing to get the final feature vector.

    def forward_once(self, image):
        # Process an image through the image network.
        image_output = self.img_net(image)
        # Transform the output to have the desired dimension.
        image_output = torch.relu(self.fc_img_1(image_output))

        # Pass the combined vector through fully connected layers.
        combined = self.fc2(image_output)

        return combined

    def forward(self, image):
        # Process inputs through the network
        output = self.forward_once(image)
        return output


def standard_vector_generation(image_folder_dir, model_dir, model_name, save_dir, required_days):
    input_dir = image_folder_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stand_save = save_dir

    if not os.path.exists(stand_save):
        os.makedirs(stand_save)
    # print(f"Using device: {device}")

    require_date = required_days

    plant_id_list = plant_id_extract(input_dir)
    real_plant_id_list = []
    temp_output_dir = "temp_outputs"
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    for plant_id in plant_id_list:
        # print(str(plant_id))
        img_list, img_date = find_files_by_days_and_id(os.listdir(input_dir), require_date, plant_id)
        # print(img_list)
        # print(img_date)
        if img_list:
            real_plant_id_list.append(plant_id)

            # model
            model_ft = FeatureExtractNetwork_no_weather(model_name=str(model_name))

            model_weights = torch.load(model_dir)

            model_ft.load_state_dict(model_weights)
            model_ft.to(device)
            model_ft.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            for img_index, img in enumerate(img_list):
                torch.cuda.empty_cache()
                img_path = os.path.join(input_dir, img)
                image = Image.open(img_path)
                image_transformed = transform(image)
                img1 = image_transformed.unsqueeze(0).to(device)

                output = model_ft(img1)

                temp_output_path = os.path.join(temp_output_dir, f"output_{plant_id}_{img_index}.pth")
                torch.save(output.cpu(), temp_output_path)

    real_plant_id_list = list(set(real_plant_id_list))

    for plant_id in real_plant_id_list:
        accumulated_outputs = torch.zeros(256, device=device)

        output_files = [f for f in os.listdir(temp_output_dir) if f.startswith(f"output_{plant_id}_")]
        for file_name in output_files:
            output_path = os.path.join(temp_output_dir, file_name)
            output = torch.load(output_path, map_location=device)
            accumulated_outputs += output.squeeze()

        average_output = accumulated_outputs / len(output_files)

        save_path = os.path.join(stand_save, f"{plant_id}.pth")

        torch.save(average_output.cpu(), save_path)

    shutil.rmtree(temp_output_dir)

    print('Done !')



def vector_generation(image_folder_dir, model_dir, model_name, save_dir):
    input_dir = image_folder_dir

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stand_save = save_dir

    # model
    model_ft = FeatureExtractNetwork_no_weather(model_name=str(model_name))

    model_weights = torch.load(model_dir)

    model_ft.load_state_dict(model_weights)
    model_ft.to(device)
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
            flowering_days = match.group(1)
            img_date = match.group(2)
        else:
            print("No match found")

        img_path = os.path.join(input_dir, img)
        image = Image.open(img_path)
        image_transformed = transform(image)
        img1 = image_transformed.unsqueeze(0).to(device)


        output = model_ft(img1)

        temp_output_path = os.path.join(stand_save, f"{os.path.splitext(img)[0]}.pth")
        torch.save(output.cpu(), temp_output_path)

        pbar.set_postfix({'img name': str(img), 'day': str(flowering_days),'date':img_date})


    print('Done !')


def find_files_not_matching_days_and_id(sample_files, days, plant_id):
    pattern = re.compile(
        fr'^{days}_([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})_(?:BC|TPA|TRF)-ID{plant_id}_IMG_\d+(_\d+)?\.pth$',
        re.IGNORECASE
    )

    non_matching_files = []

    for file in sample_files:
        match = pattern.match(file)
        if not match:

            non_matching_files.append(file)

    return non_matching_files


def generate_unique_samples(stand_id_list, group_size, sample_size):
    stand_list_group = []
    attempts = 0

    while len(stand_list_group) < group_size and attempts < 100:
        attempts += 1
        candidate = random.sample(stand_id_list, min(sample_size, len(stand_id_list)))


        valid = True
        for i in range(len(stand_list_group)):
            if candidate[0] == stand_list_group[i][0]:
                valid = False
                break
            if sample_size >= 3 and candidate[:3] == stand_list_group[i][:3]:
                valid = False
                break
            if sample_size >= 5 and candidate[:5] == stand_list_group[i][:5]:
                valid = False
                break

        if valid:
            stand_list_group.append(candidate)

    while len(stand_list_group) < group_size:
        candidate = random.sample(stand_id_list, min(sample_size, len(stand_id_list)))
        stand_list_group.append(candidate)

    return stand_list_group


home_dir = 'I:/wheat project/few shot flowering project/paper'
img_folder_dir = os.path.join(home_dir, 'early dataset image')
model_save_dir = 'BC convnext'
model_save_folder = 'model no weather'
model_dir = os.path.join(home_dir, model_save_folder, model_save_dir, 'feature extraction model.pth')
sim_model_dir = os.path.join(home_dir, model_save_folder, model_save_dir,'compare model.pth')

model_name = 'convnex'
epoch = 'best'
few_shot_num_list = [1,3,5]

save_dir = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), 'early dataset  result')
stand_save_model_ft = os.path.join(home_dir, model_save_folder,model_save_dir,str(epoch), 'early dataset  output 1')
stand_save_total = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), 'early dataset  stand vector save')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model_ft
if not os.path.exists(stand_save_model_ft) or not os.path.isdir(stand_save_model_ft):
    vector_generation(img_folder_dir,model_dir, model_name,stand_save_model_ft)
#
# # model standard vector
required_days_list = [12]

if not os.path.exists(stand_save_total) or not os.path.isdir(stand_save_total):
    for required_days in required_days_list:
        print('Require_date stand vector: ', str(required_days))
        stand_save = os.path.join(stand_save_total, str(required_days))
        if not os.path.exists(stand_save):
            os.makedirs(stand_save)
        standard_vector_generation(img_folder_dir, model_dir, model_name, stand_save, required_days)


if 'tf' in model_save_dir:
    sim_model = ComparedNetwork_Transformer()
else:
    sim_model = ComparedNetwork()
sim_model_weights = torch.load(sim_model_dir)
sim_model.load_state_dict(sim_model_weights)
sim_model.to(device)
sim_model.eval()

print(f"Start: Using device: {device}")


for require_date in required_days_list:
    print('Require_date: ', str(require_date))
    # stand_dir_seed = os.path.join(stand_save_total, str(require_date))
    # stand_id_list = sorted(os.listdir(stand_dir_seed))
    # # stand_list_group = []
    # #
    # # for _ in range(10):
    # #     stand_list = random.sample(stand_id_list, min(10, len(stand_id_list)))
    # #     stand_list_group.append(stand_list)
    # stand_list_group = generate_unique_samples(stand_id_list, 10, 5)
    transformed_list = generate_unique_ID(img_folder_dir, set(range(8, 17)))

    few_shot_eval(stand_save_model_ft, stand_save_total, save_dir, 512, sim_model, require_date, few_shot_num_list,device,transformed_list)
    print('--------------------------------------------------')