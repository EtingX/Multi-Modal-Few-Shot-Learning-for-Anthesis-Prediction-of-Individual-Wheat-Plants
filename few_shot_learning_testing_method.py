import os
import random
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from model_structure import *
import shutil
def find_files_not_matching_days_and_id(sample_files, days, plant_id):
    """
    在提供的文件列表中寻找不符合特定天数和植物ID的文件。

    :param sample_files: 文件名列表。
    :param days: 指定的天数。
    :param plant_id: 指定的植物ID。
    :return: 不匹配的文件名列表。
    """
    # 构建正则表达式以匹配指定格式的文件名，并捕获日期部分
    # 更新的正则表达式
    pattern = re.compile(
        fr'^{days}_([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})_(?:BC|TPA|TRF)-ID{plant_id}_IMG_\d+(_\d+)?\.pth$',
        re.IGNORECASE
    )

    non_matching_files = []

    for file in sample_files:
        match = pattern.match(file)
        if not match:
            # 如果文件不匹配，则添加到结果列表中
            non_matching_files.append(file)

    return non_matching_files


def few_shot_eval(test_path, stand_dir, save_path, batch_size, sim_model, require_date,
                  few_shot_num_list, device, stand_list_group):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    required_days = str('required day-' + str(require_date))
    save_dir = os.path.join(save_path, required_days)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    stand_dir = os.path.join(stand_dir,str(require_date))
    stand_id_list = os.listdir(stand_dir)

    # print('Few shot list: ' + str(few_shot_num_list))
    for few_shot_num in few_shot_num_list:
        stand_list_few_shot_num_group = [sublist[:few_shot_num] for sublist in stand_list_group]
        print(stand_list_few_shot_num_group)
        all_F1 = []
        select_id_list = []
        # print('Few shot num: ' + str(few_shot_num))

        if len(stand_id_list) <= 10 and few_shot_num == 1:
            random_num = len(stand_id_list)
        else:
            random_num = 10
        for n in range(random_num):
            # print('This is ' + str(n) + ' run.')
            if len(stand_id_list) <= random_num and few_shot_num == 1:
                stand_list = [stand_id_list[n]]
            else:
                stand_list = stand_list_few_shot_num_group[n]

            # print(stand_list)
            accumulated_outputs = torch.zeros(256, device=device)
            img_list = os.listdir(test_path)
            # print('original img number: ' + str(len(img_list)))
            # if 'BC' in os.path.basename(test_path):
            #     img_list = [f for f in img_list if any(f.startswith(str(i) + '_') for i in range(6, 15))]
            # elif 'TPA' in os.path.basename(test_path):
            #     img_list = [f for f in img_list if any(f.startswith(str(i) + '_') for i in range(6, 19))]
            # print('filtered img number: ' + str(len(img_list)))
            for stand_id in stand_list:
                match = re.match(r"(\d+).pth", stand_id)
                stand_id_num = match.group(1)
                # print('Stand plant id: ' + str(stand_id_num))
                output_path = os.path.join(stand_dir, stand_id)
                output = torch.load(output_path, map_location=device)  # 确保加载到正确的设备上
                accumulated_outputs += output.squeeze()
                img_list = find_files_not_matching_days_and_id(img_list, str(require_date), str(stand_id_num))
            # if n == 0:
            #     print(img_list)

            # 处理stand_list来作为文件名的一部分
            stand_names_excel = '_'.join([s.split('.')[0] for s in stand_list])

            average_output = accumulated_outputs / len(stand_list)

            # print(average_output.size())
            # temp_output_path = "two_class_output2_tmp.pth"
            # torch.save(average_output.cpu(), temp_output_path)

            dataset_test = CustomDataset_few_shot(test_path, img_list, require_date)

            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

            true_label_list = []
            predict_label_list = []
            img_id_list = []
            with torch.no_grad():
                for data in tqdm(test_loader, desc="Processing Images"):
                    img_name, output1, target = data

                    for i in img_name:
                        img_id_list.append(i)

                    for t in target:
                        true_label_list.append(t.data.item())

                    output1 = output1.to(device, non_blocking=True)
                    average_output = average_output.to(device)
                    output1 = output1.view(output1.size(0), -1)
                    current_batch_size = output1.size(0)

                    average_output_adj = average_output.unsqueeze(0).repeat(current_batch_size, 1)
                    output = sim_model(output1, average_output_adj)
                    _, pred = torch.max(output.data, 1)
                    for p in pred:
                        predict_label_list.append(p.data.item())

            # 删除Tensor
            del average_output
            del average_output_adj

            # 清空CUDA缓存，释放内存
            torch.cuda.empty_cache()

            img_id_list = [img_id.replace('.pth', '') for img_id in img_id_list]
            # 创建DataFrame
            df_run = pd.DataFrame({
                'Image ID': img_id_list,
                'Predicted Label': predict_label_list,
                'True Label': true_label_list
            })

            # 定义文件名
            filename = f"{stand_names_excel}.xlsx"

            if not os.path.exists(os.path.join(save_dir,str(few_shot_num))):
                os.makedirs(os.path.join(save_dir,str(few_shot_num)))
            # 确保文件路径是正确的，这里只是一个示例
            excel_path_stand = os.path.join(save_dir,str(few_shot_num), filename)  # 你可能需要根据实际路径调整

            # 保存到Excel
            df_run.to_excel(excel_path_stand, index=False)



            F1 = f1_score(true_label_list, predict_label_list, average='weighted')

            # 生成混淆矩阵
            cm = confusion_matrix(true_label_list, predict_label_list)
            # print(f"Saved to {excel_path_stand}")
            # print(classification_report(true_label_list, predict_label_list))
            # print("Confusion Matrix:")
            # print(cm)

            all_F1.append(F1)
            select_id_list.append(stand_list)

        df = pd.DataFrame({
            'Select ID List': select_id_list,
            'F1 Score': np.round(all_F1, 5)  # 保留5位小数
        })

        # 在所有迭代完成后
        all_F1_array = np.array(all_F1)
        average_all_F1 = np.mean(all_F1_array)  # 计算所有F1分数的平均值
        print('Few shot num: ' + str(few_shot_num) + ' result')
        print("F1 Scores:", all_F1_array)  # 打印所有F1分数
        print("Average F1 Score:", average_all_F1)  # 打印平均F1分数

        df.loc[len(df)] = ['Average', np.round(average_all_F1, 5)]  # 添加平均行

        # 使用with语句和ExcelWriter来自动保存和关闭文件
        results_file_path = os.path.join(save_dir, f'results_{few_shot_num}_shot.xlsx')
        with pd.ExcelWriter(results_file_path, engine='xlsxwriter') as writer:
            sheet_name_few = f'{few_shot_num} shot'
            df.to_excel(writer, sheet_name=sheet_name_few, index=False)

        # print(f'Results for {few_shot_num} shots saved!')


from collections import defaultdict

def generate_unique_samples(stand_id_list, group_size, sample_size):
    random.seed(1993)
    stand_list_group = []
    attempts = 0
    while len(stand_list_group) < group_size and attempts < 100:  # 限制尝试次数避免死循环
        attempts += 1
        candidate = random.sample(stand_id_list, min(sample_size, len(stand_id_list)))

        # 检查前1位，前3位，前5位是否有重复
        valid = True
        for i in range(len(stand_list_group)):
            print(stand_list_group)
            if candidate[0] == stand_list_group[i][0]:  # 检查前1位
                valid = False
                break
            if sample_size >= 3 and candidate[:3] == stand_list_group[i][:3]:  # 检查前3位
                valid = False
                break
            if sample_size >= 5 and candidate[:5] == stand_list_group[i][:5]:  # 检查前5位
                valid = False
                break

        if valid:
            stand_list_group.append(candidate)

    # 强制补全至 group_size
    while len(stand_list_group) < group_size:
        candidate = random.sample(stand_id_list, min(sample_size, len(stand_id_list)))
        stand_list_group.append(candidate)

    return stand_list_group

def generate_unique_ID(dir, day_range):
    filenames = os.listdir(dir)
    id_days = defaultdict(set)
    pattern = re.compile(r'^(\d+)_\d{4}-\d{2}-\d{2}_(TPA|TRF|BC)-ID(\d+)_')
    print(filenames)
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            day, image_id = int(match.group(1)), int(match.group(3))  # 正确使用第三个分组获取ID
            if day in day_range:
                id_days[image_id].add(day)

    # Filter IDs that have all days from day_range and create a list of valid IDs
    valid_ids = sorted([image_id for image_id, days in id_days.items() if day_range <= days])
    print(valid_ids)
    stand_list_group = generate_unique_samples(valid_ids, 10, 15)

    transformed_list = [[f"{num}.pth" for num in sublist] for sublist in stand_list_group]

    return transformed_list



