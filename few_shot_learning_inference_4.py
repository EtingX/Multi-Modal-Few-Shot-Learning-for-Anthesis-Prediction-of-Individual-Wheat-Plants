from few_shot_learning_evaluate_standard_anchor import *
from few_shot_learning_testing_method import *
from few_show_vector_generation import *


home_dir = r'I:\wheat project\few shot flowering project\paper\share'
weather_folder_dir = os.path.join(home_dir, 'early dataset weather')
model_save_folder = 'model'
model_save_dir_list = os.listdir(os.path.join(home_dir,model_save_folder))
print(model_save_dir_list)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for model_save_dir in model_save_dir_list:
    print(model_save_dir)
    model_dir = os.path.join(home_dir, model_save_folder, model_save_dir, 'feature extraction model.pth')
    sim_model_dir = os.path.join(home_dir, model_save_folder, model_save_dir, 'compare model.pth')

    # how much shot. one-plant, three-plant, five-plant
    few_shot_num_list = [1, 5, 10]
    if 'convnext' in model_save_dir:
        model_name = 'convnext'
    else:
        model_name = 'swin_b'
    epoch = 'best'
    if 'tf' in model_save_dir:
        sim_model = ComparedNetwork_Transformer()
    else:
        sim_model = ComparedNetwork()

    #######################################################################################################
    img_folder_dir = os.path.join(home_dir, 'early dataset image')
    save_dir = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), 'early dataset 2023 result')
    stand_save_model_ft = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), 'early dataset  2023 output 1')
    stand_save_total = os.path.join(home_dir, model_save_folder, model_save_dir, str(epoch), 'early dataset stand vector save')


    if not os.path.exists(stand_save_model_ft) or not os.path.isdir(stand_save_model_ft):
        # model_ft, metal learning feature extraction
        vector_generation(img_folder_dir, weather_folder_dir, model_dir, model_save_dir, stand_save_model_ft)
    # #
    # # # model standard vector
    required_days_list = [8, 16]
    #
    for required_days in required_days_list:
        print('Require_date stand vector: ', str(required_days))
        stand_save = os.path.join(stand_save_total, str(required_days))
        # anchor building
        if not os.path.exists(stand_save):
            os.makedirs(stand_save)
            standard_vector_generation(img_folder_dir, weather_folder_dir, model_dir, model_name, stand_save, required_days)

    sim_model_weights = torch.load(sim_model_dir)
    sim_model.load_state_dict(sim_model_weights)
    sim_model.to(device)
    sim_model.eval()
    # print(img_folder_dir)
    # for anchor fix testing
    transformed_list = generate_unique_ID(stand_save_model_ft, set(range(8, 17))) # put what days range before flower ID you want

    print(f"Start: Using device: {device}")
    for require_date in required_days_list:
        print('Require_date: ', str(require_date))
        few_shot_eval(stand_save_model_ft, stand_save_total, save_dir, 512, sim_model, require_date, few_shot_num_list,device, transformed_list)
        print('--------------------------------------------------')
