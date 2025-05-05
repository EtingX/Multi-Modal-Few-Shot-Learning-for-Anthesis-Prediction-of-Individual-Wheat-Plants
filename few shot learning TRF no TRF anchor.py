import os.path

from few_shot_learning_evaluate_standard_anchor import *
from few_shot_learning_testing_method import *
from few_show_vector_generation import *

def generate_unique_samples(stand_id_list, group_size, sample_size):
    stand_list_group = []
    attempts = 0

    while len(stand_list_group) < group_size and attempts < 100:  # Limit the number of attempts to prevent infinite loops.
        attempts += 1
        candidate = random.sample(stand_id_list, min(sample_size, len(stand_id_list)))

        # Check for duplicates in the first 1, 3, and 5 ID.
        valid = True
        for i in range(len(stand_list_group)):
            if candidate[0] == stand_list_group[i][0]:  # 1
                valid = False
                break
            if sample_size >= 3 and candidate[:3] == stand_list_group[i][:3]:  # 3
                valid = False
                break
            if sample_size >= 5 and candidate[:5] == stand_list_group[i][:5]:  # 5
                valid = False
                break

        if valid:
            stand_list_group.append(candidate)

    # Force completion up to the group size
    while len(stand_list_group) < group_size:
        candidate = random.sample(stand_id_list, min(sample_size, len(stand_id_list)))
        stand_list_group.append(candidate)

    return stand_list_group

seed_num = 1993

home_dir = r'I:\wheat project\few shot flowering project\paper\share'
img_folder_dir = os.path.join(home_dir, 'TRF 2023 image')
weather_folder_dir = os.path.join(home_dir, 'TRF dataset weather')
model_save_folder = 'model'
#
model_save_dir_list = ['BC convnext_tf', 'TPA 1 convnext_tf', 'TPA 2 convnext_tf',
                       'BC convnext', 'TPA 1 convnext', 'TPA 2 convnext']

for model_save_dir in model_save_dir_list:

    model_dir = os.path.join(home_dir, model_save_folder, model_save_dir, 'feature extraction model.pth')
    sim_model_dir = os.path.join(home_dir, model_save_folder, model_save_dir, 'compare model.pth')

    model_name = 'convnext'
    epoch = 'best'
    few_shot_num_list = [1, 3, 5]
    stand_save_model_ft = os.path.join(home_dir, model_save_folder, model_save_dir,str(epoch), 'TRF 2023 output 1')
    if not os.path.exists(stand_save_model_ft) or not os.path.isdir(stand_save_model_ft):
        vector_generation(img_folder_dir, weather_folder_dir, model_dir, model_save_dir, stand_save_model_ft)

    stand_save_total_list = ['BC 2023', 'TPA 1 2023', 'TPA 2 2023']

    for new_stand in stand_save_total_list:
        save_dir = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), 'TRF 2023 result ' + str(new_stand))
        stand_save_total = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), new_stand+' stand vector save')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # # # model standard vector
        required_days_list = [8]
        #
        if not os.path.exists(stand_save_total) or not os.path.isdir(stand_save_total):
            for required_days in required_days_list:
                print('Require_date stand vector: ', str(required_days))
                stand_save = os.path.join(stand_save_total, str(required_days))
                if not os.path.exists(stand_save):
                    os.makedirs(stand_save)
                    if new_stand == 'BC 2023':
                        standard_vector_generation(os.path.join(home_dir, 'late dataset image')
                                                   , weather_folder_dir, model_dir, model_name, stand_save, required_days)
                    elif new_stand == 'TPA 1 2023':
                        standard_vector_generation(os.path.join(home_dir, 'early dataset image')
                                                   , weather_folder_dir, model_dir, model_name, stand_save, required_days)
                    elif new_stand == 'TPA 2 2023':
                        standard_vector_generation(os.path.join(home_dir, 'mid dataset image')
                                                   , weather_folder_dir, model_dir, model_name, stand_save, required_days)


        if 'tf' in model_save_dir:
            sim_model = ComparedNetwork_Transformer()
        else:
            sim_model = ComparedNetwork()
        sim_model_weights = torch.load(sim_model_dir)
        sim_model.load_state_dict(sim_model_weights)
        sim_model.to(device)
        sim_model.eval()

        print(f"Start: Using device: {device}")
        random.seed(seed_num)
        for require_date in required_days_list:
            print('Require_date: ', str(require_date))
            stand_dir_seed = os.path.join(stand_save_total, str(require_date))
            stand_id_list = sorted(os.listdir(stand_dir_seed))
            # stand_list_group = []
            #
            # for _ in range(10):
            #     stand_list = random.sample(stand_id_list, min(10, len(stand_id_list)))
            #     stand_list_group.append(stand_list)
            stand_list_group = generate_unique_samples(stand_id_list, 10, 5)
            few_shot_eval(stand_save_model_ft, stand_save_total, save_dir, 512, sim_model, require_date, few_shot_num_list,device, stand_list_group)
            print('--------------------------------------------------')