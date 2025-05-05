import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report, f1_score, confusion_matrix
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
from model_structure import *


# 验证过程
@torch.no_grad()
def val(model, sim_model, device, test_loader):
    # global Best_ACC
    global Best_F1
    model.eval()
    sim_model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))

    # Stores true and predict labels
    val_list = []
    pred_list = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Evaluation process')
    # Iterate over the test dataset
    for batch_idx, data in pbar:
        img1, weather1, img2, weather2, target = data
        for t in target:
            val_list.append(t.data.item())
        img1, weather1 = img1.to(device, non_blocking=True), weather1.to(device, non_blocking=True)
        img2, weather2 = img2.to(device, non_blocking=True), weather2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Generate predictions
        output1 = model((img1, weather1))
        output2 = model((img2, weather2))
        output = sim_model(output1, output2)

        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())

        # Update accuracy and loss trackers, only acc1 is useful as only two classes
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        pbar.set_postfix({'Loss': loss_meter.avg})

    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc5_meter.avg))

    # Compute F1 score
    F1 = f1_score(val_list, pred_list, average='weighted')

    return val_list, pred_list, loss_meter.avg, acc, F1


if __name__ == '__main__':
    '''
    This script is designed to evaluate the model performance on the different dataset.
    Hint: choose correct model with correct net in the image net. metal learning model and comparable model.
    'tf' in sim_model_path means comparable model use transformer use # model_similarity = ComparedNetwork_Transformer()
    '''
    architecture = 'convnext'
    model_ft = FeatureExtractNetwork(model_name=str(architecture))
    model_similarity = ComparedNetwork()
    # model_similarity = ComparedNetwork_Transformer()

    test_path = 'I:/wheat project/few shot flowering project/few shot data set/new_train_data/BC 2023 dataset new/val'
    # model pathway
    model_path = r"I:\wheat project\few shot flowering project\paper\share\model\BC convnext\feature extraction model.pth"
    sim_model_path = r"I:\wheat project\few shot flowering project\paper\share\model\BC convnext\compare model.pth"

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion_val = torch.nn.CrossEntropyLoss()

    model_ft.to(DEVICE)  # Move model to the correct device
    model_similarity.to(DEVICE)  # Move model to the correct device

    model_ft.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_similarity.load_state_dict(torch.load(sim_model_path, map_location=DEVICE))

    dataset_test = CustomDataset(test_path)
    print('Evaluation start !')
    print('Dataset path: ' + str(test_path))
    print(str(model_path))
    print((str(sim_model_path)))

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=2)

    val_list, pred_list, val_loss, val_acc, F1 = val(model_ft, model_similarity, DEVICE, test_loader)

    print('F1 score: ' + str(round(F1, 4)))
    print(classification_report(val_list, pred_list))

    # 生成混淆矩阵
    cm = confusion_matrix(val_list, pred_list)

    print("Confusion Matrix:")
    print(cm)

