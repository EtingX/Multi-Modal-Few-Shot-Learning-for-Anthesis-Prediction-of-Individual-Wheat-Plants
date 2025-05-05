import torch.optim as optim
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
from model_structure import *
import shutil

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print_val_loss_min = self.val_loss_min * -1
            print_val_loss = val_loss * -1  # -1 because using F1 score, if loss change to 1
            print(f'F1 score increased ({print_val_loss_min:.6f} --> {print_val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')  # save model here

        self.val_loss_min = val_loss


class EMA():
    '''
        strategy can be particularly useful for large models or when aiming
        to improve the stability and generalization of the training process.
    '''

    def __init__(self, model, decay):
        # Initializing the EMA class with a model and a decay rate.
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        # Registering the model's parameters with ema.register() before the training loop.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # Updating the EMA of the parameters after each epoch or update step.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # Applying the EMA parameters for evaluation or inference using ema.apply_shadow().
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # Restoring the original parameters with ema.restore() after evaluation to
        # continue training or for other purposes.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def label_smoothing(labels, classes=2, epsilon=0.1):
    """
    Applies label smoothing. Label smoothing is a regularization technique used in training classification models.
    It modifies the hard labels to be slightly less confident (closer to uniform distribution), which can help the
    model generalize better to unseen data by preventing it from becoming too confident about its predictions.

    Parameters:
    - labels (torch.Tensor): The original labels with shape [N], where N is the batch size.
    - classes (int): The total number of classes. Default is 2.
    - epsilon (float): The smoothing parameter.

    Returns:
    - torch.Tensor: The smoothed labels with shape [N, classes].
    """
    N = labels.size(0)
    smoothed_labels = torch.full(size=(N, classes), fill_value=epsilon / classes, device=labels.device)
    smoothed_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon + (epsilon / classes))
    return smoothed_labels


def train(model, sim_model, device, train_loader, optimizer, epoch):
    """
    Trains the model for one epoch.

    Parameters:
    - model: The feature extraction model.
    - sim_model: The model that computes similarities between pairs of features.
    - device: The device (CPU or GPU) to train on.
    - train_loader: DataLoader for training data.
    - optimizer: The optimizer.
    - epoch: Current epoch number.
    - use_amp: Flag to use automatic mixed precision.
    - use_ema: Flag to use exponential moving average.
    - criterion_train: Loss function.
    - accuracy: Function to compute accuracy metrics.
    - ema: EMA object if use_ema is True.

    Returns:
    - Average loss and accuracy for the epoch.
    """

    model.train()
    sim_model.train()

    # Metrics class
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(f'Total number of samples: {total_num}, Batches: {len(train_loader)}')

    # Progress bar with tqdm
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, data in pbar:
        img1, weather1, img2, weather2, target = data
        img1, weather1 = img1.to(device, non_blocking=True), weather1.to(device, non_blocking=True)
        img2, weather2 = img2.to(device, non_blocking=True), weather2.to(device, non_blocking=True)
        targets = target.to(device, non_blocking=True)
        target = label_smoothing(target, epsilon=0.1).to(device, non_blocking=True)
        output1 = model((img1, weather1))
        output2 = model((img2, weather2))
        output = sim_model(output1, output2)
        # print(output)

        optimizer.zero_grad()
        if use_amp:
            # Automatic mixed precision
            with torch.cuda.amp.autocast():
                loss = criterion_train(output, target)
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            # Apply EMA model parameters for evaluation if enabled
            if use_ema:
                ema.update()
        else:
            loss = criterion_train(output, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            # Apply EMA model parameters for evaluation if enabled
            if use_ema:
                ema.update()

        # Synchronize GPU operations if necessary
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), targets.size(0))

        # Update metrics, only acc1 is useful, as only two classes
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        # Update progress bar
        pbar.set_postfix({'Loss': loss_meter.avg, 'LR': lr})

    print(f'Epoch: {epoch}\tLoss: {loss_meter.avg:.4f}\tAcc: {acc1_meter.avg:.4f}')
    return loss_meter.avg, acc1_meter.avg


# Val section
@torch.no_grad()
def val(model, sim_model, device, test_loader):
    """
        Evaluates the model on a test dataset.

        Parameters:
        - model (torch.nn.Module): The main model for feature extraction.
        - sim_model (torch.nn.Module): A similarity comparison model that operates on the features extracted
        by the main model.
        - device (torch.device): The device to perform the evaluation on (e.g., 'cuda' or 'cpu').
        - test_loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.

        Returns:
        - val_list (list): A list of true labels from the test dataset.
        - pred_list (list): A list of predicted labels for the test dataset.
        - loss_meter.avg (float): The average loss computed across all test data.
        - acc (float): The top-1 accuracy percentage across the test dataset.
        - F1 (float): The weighted F1 score based on true and predicted labels.

        This function performs evaluation by iterating through the test dataset, computing predictions and loss, and
        then calculating accuracy metrics including F1 score.

        """
    # global Best_ACC
    global Best_F1 # Tracks the best F1 score across epochs
    model.eval()
    sim_model.eval()

    # Metrics class
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))

    # Stores true and predict labels
    val_list = []
    pred_list = []

    # Apply EMA model parameters for evaluation if enabled
    if use_ema:
        ema.apply_shadow()

    # Iterate over the test dataset
    for data in test_loader:
        img1, weather1, img2, weather2, target = data

        # Record true labels
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

        # Determine predicted classes
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())

        # Update accuracy and loss trackers, only acc1 is useful as only two classes
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

    # Restore original model parameters if EMA was applied
    if use_ema:
        ema.restore()
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc5_meter.avg))

    # Compute F1 score
    F1 = f1_score(val_list, pred_list, average='weighted')

    # Check for F1 improvement and save models accordingly
    if F1 > Best_F1 and F1 >= 0.75:
        wandb.save('best_F1_model.h5')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.state_dict(), file_dir + "/" + 'model_F1_' + str(epoch) + '_' + str(round(F1, 3)) + '.pth')
            torch.save(model.state_dict(), file_dir + '/' + 'F1_best.pth')
            torch.save(sim_model.state_dict(),
                       file_dir + "/" + 'sim_model_F1_' + str(epoch) + '_' + str(round(F1, 3)) + '.pth')
            torch.save(sim_model.state_dict(), file_dir + '/' + 'sim_F1_best.pth')
        else:
            torch.save(model.state_dict(), file_dir + "/" + 'model_F1_' + str(epoch) + '_' + str(round(F1, 3)) + '.pth')
            torch.save(model.state_dict(), file_dir + '/' + 'F1_best.pth')
            torch.save(sim_model.state_dict(),
                       file_dir + "/" + 'sim_model_F1_' + str(epoch) + '_' + str(round(F1, 3)) + '.pth')
            torch.save(sim_model.state_dict(), file_dir + '/' + 'sim_F1_best.pth')
        Best_F1 = F1

    # Save the last model state
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.state_dict(), file_dir + "/" + 'last.pth')
        torch.save(sim_model.state_dict(), file_dir + "/" + 'sim_last.pth')
    else:
        torch.save(model.state_dict(), file_dir + "/" + 'last.pth')
        torch.save(sim_model.state_dict(), file_dir + "/" + 'sim_last.pth')

    return val_list, pred_list, loss_meter.avg, acc, F1


# 修改sgd_optimizer函数以接受named_parameters列表
def sgd_optimizer(named_params, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in named_params:
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            # Just a Caffe-style common practice. Made no difference.
            apply_lr = 2 * lr
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


# 修改adam_optimizer函数以接受named_parameters列表
def adam_optimizer(named_params, lr, weight_decay, use_custwd):
    params = []
    for key, value in named_params:
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            # Just a Caffe-style common practice. Made no difference.
            apply_lr = 2 * lr
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.Adam(params, lr)
    return optimizer


if __name__ == '__main__':
    '''
    Login to the wandb first and then train the whole process, the RepVgg is RepVGG_B3
    Changing option could be 'optimizer' and 'net architecture' 
    All models are in Model_YX.py
    '''

    project = 'BC full 2023_wheat_two_classification'
    run_name = 'swin_v2_s tf 1 4 max'
    config = dict(
        learning_rate=1e-3,
        eta_min=1e-5,
        final_learning_rate=1e-5,
        EPOCHS=60,
        architecture="swin_v2_s tf 1 4 max real",
        optimizer='ADAM',
        opt_momentum=0.9,
        opt_weight_decay=1e-4,
        train_batch_size=64,
        test_batch_size=64,
        DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        use_amp=True,  # 是否使用混合精度
        use_dp=False,  # multiple GPU
        classes=['0', '1'],
        resume=False,
        use_ema=True,
        train_loss='SoftTargetCrossEntropy',
        val_loss='CrossEntropy',
        attention=False,
        early_stop=True,
        schedule_lr='cosine'
    )

    wandb.init(project=project,
               name=run_name,
               config=config,
               mode='online')
    wandb.save("*.pt")
    wandb.watch_called = False

    # 设置全局参数
    model_lr = wandb.config.learning_rate
    train_BATCH_SIZE = wandb.config.train_batch_size
    test_BATCH_SIZE = wandb.config.test_batch_size
    optimizer = wandb.config.optimizer
    EPOCHS = wandb.config.EPOCHS
    DEVICE = wandb.config.DEVICE
    use_amp = wandb.config.use_amp
    use_dp = wandb.config.use_dp
    classes = len(wandb.config.classes)
    resume = wandb.config.resume
    use_ema = wandb.config.use_ema
    early_stop = wandb.config.early_stop
    schedule_lr = wandb.config.schedule_lr
    architecture = wandb.config.architecture

    print(DEVICE)
    model_path = ''
    mode_sim_path = ''
    # record highest result (established)

    Best_F1 = 0

    patience = 10
    early_stopping = EarlyStopping(patience, verbose=True)

    # save model dir
    localtime = time.asctime(time.localtime(time.time())).split()
    str_time = str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3][0:2])
    # checkpoint folder
    file_dir = '/content/drive/MyDrive/wheat_few_shot_project/BC 2023 full_save_two_class/' + str(
        wandb.config.architecture) + '/' + str_time
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)

    txt = file_dir + '/' + 'config.txt'
    filename = open(txt, 'w')  # dict to txt
    for k, v in config.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()

    # # 随机分割数据集
    train_path = '/content/drive/MyDrive/wheat_few_shot_project/BC 2023 full/train'
    test_path = '/content/drive/MyDrive/wheat_few_shot_project/BC 2023 full/val'

    print(train_path)
    print(test_path)

    dataset_train = CustomDataset(train_path)
    dataset_test = CustomDataset(test_path)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=train_BATCH_SIZE, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_BATCH_SIZE, shuffle=False, num_workers=16)

    # criterion to GPU
    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()
    # 初始化模型、优化器和损失函数
    model_ft = FeatureExtractNetwork(model_name=str(architecture))
    model_similarity = ComparedNetwork_Transformer()
    # model_similarity = ComparedNetwork()
    print(model_ft)
    print(model_similarity)

    if resume:
        model_ft.load_state_dict(torch.load(model_path))
        model_similarity.load_state_dict(torch.load(mode_sim_path))
    model_ft.to(DEVICE)
    model_similarity.to(DEVICE)

    wandb.watch(model_ft, log="all")

    # if choose Adam, reduce learning rate
    if optimizer == 'SGD' or 'sgd':
        combined_params = list(model_ft.named_parameters()) + list(model_similarity.named_parameters())
        print(optimizer)
        optimizer = sgd_optimizer(combined_params, model_lr, momentum=wandb.config.opt_momentum,
                                  weight_decay=wandb.config.opt_weight_decay, use_custwd=False)

    elif optimizer == 'ADAM' or 'Adam':
        combined_params = list(model_ft.named_parameters()) + list(model_similarity.named_parameters())
        print(optimizer)
        optimizer = adam_optimizer(combined_params, model_lr, weight_decay=wandb.config.opt_weight_decay,
                                   use_custwd=False)

    elif optimizer == 'ADAM_normal' or 'Adam_normal':
        combined_params = list(model_ft.parameters()) + list(model_ft.parameters())
        print(optimizer)
        optimizer = optim.Adam(combined_params, lr=model_lr)
    elif optimizer == 'SGD_normal' or 'sgd_normal':
        combined_params = list(model_ft.parameters()) + list(model_ft.parameters())
        print(optimizer)
        optimizer = optim.sgd(combined_params, lr=model_lr)

    if schedule_lr == 'cosine':
        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20,
                                                        eta_min=wandb.config.eta_min)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = torch.nn.DataParallel(model_ft)

    if use_ema:
        ema = EMA(model_ft, 0.999)
        ema.register()

    is_set_lr = False
    # train and val start
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model_ft, model_similarity, DEVICE, train_loader, optimizer, epoch)
        wandb.log({
            "Epoch": epoch,
            "Train Accuracy": train_acc,
            "Train Loss": train_loss
        })

        val_list, pred_list, val_loss, val_acc, F1 = val(model_ft, model_similarity, DEVICE, test_loader)

        if early_stopping and epoch >= 10:
            early_stopping(-F1, model_ft)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break

        print(classification_report(val_list, pred_list))
        # 生成混淆矩阵
        cm = confusion_matrix(val_list, pred_list)

        print("Confusion Matrix:")
        print(cm)

        wandb.log({
            "Epoch": epoch,
            "Val Accuracy": val_acc,
            "Val Loss": val_loss,
            'Val F1': F1
        })

        if epoch < 50:
            schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = wandb.config.final_learning_rate
                    is_set_lr = True