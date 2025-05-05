import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, \
    convnext_base,ConvNeXt_Base_Weights, swin_v2_b, Swin_V2_B_Weights
import time
import wandb
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report, f1_score, confusion_matrix
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")


class CustomDataset(Dataset):
    '''
    This CustomDataset class is designed for PyTorch and is tailored to handle datasets where each sample consists of
    pairs of images along with their corresponding weather data and a label.
    '''
    def __init__(self, folder_path):
        # Initialize dataset with path to data
        self.folder_path = folder_path
        # List all file paths in the folder, filtering out directories
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                      os.path.isfile(os.path.join(folder_path, f))]

        # Define transformations for the images
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Return the total number of files
        return len(self.files)

    def __getitem__(self, idx):
        # Load data from the file at the given index
        data = np.load(self.files[idx], allow_pickle=True)
        image_data1, weather_data1 = data[0]
        image_data2, weather_data2 = data[1]

        image1 = self.image_transform(image_data1)
        image2 = self.image_transform(image_data2)

        # Convert weather data from DataFrame to NumPy array if needed
        if isinstance(weather_data1, pd.DataFrame):
            weather_data1 = weather_data1.values
        if isinstance(weather_data2, pd.DataFrame):
            weather_data2 = weather_data2.values
        # Extract label from filename, assuming it's the first part of the filename before an underscore
        # more label strategy in the data_maker_two_class_4.py
        label = int(os.path.basename(self.files[idx]).split('_')[0])

        return image1, torch.tensor(weather_data1, dtype=torch.float), \
            image2, torch.tensor(weather_data2, dtype=torch.float), \
            torch.tensor(label, dtype=torch.long)



class CustomDataset_few_shot(Dataset):
    '''
    This CustomDataset class for few shot is designed for PyTorch and is tailored to handle datasets where each sample consists of
    pairs of images along with their corresponding weather data and a label.
    '''
    def __init__(self, folder_path, img_list, required_days):
        # Initialize dataset with path to data
        self.folder_path = folder_path
        self.img_list = img_list
        self.required_days = required_days


    def __len__(self):
        # Return the total number of files
        return len(self.img_list)

    def __getitem__(self, idx):
        # Load data from the file at the given index
        img = self.img_list[idx]
        vector_1 = str(os.path.splitext(img)[0]) + '.pth'
        output_path = os.path.join(self.folder_path, vector_1)
        output1 = torch.load(output_path)

        # Extract label from filename, assuming it's the first part of the filename before an underscore
        # more label strategy in the data_maker_two_class_4.py
        label = int(os.path.basename(img).split('_')[0])

        # 计算与14天的差值 sample 1 < sample 2
        label = 1 if label < self.required_days else 0

        return str(os.path.splitext(img)[0]), output1, torch.tensor(label, dtype=torch.long)


class FeatureExtractNetwork(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Image network - Choose one based on requirement. Uncomment the needed line.
        if 'swin_b' in model_name:
            self.img_net = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        elif 'convnext' in model_name:
            self.img_net = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            print('Please provide correct model name, efficientnet or swin')

        # RNN for processing sequential data, e.g., weather information.
        self.rnn = nn.GRU(input_size=6, hidden_size=512, num_layers=3, batch_first=True)
        # 7 add after sowing days
        # self.rnn = nn.GRU(input_size=7, hidden_size=512, num_layers=3, batch_first=True)
        self.fc_img_1 = nn.Linear(1000, 512)

        # Fully connected layers for processing combined features from the image and weather data.
        self.fc1 = nn.Linear(512 + 512, 512)  # Combining image and weather features.
        self.fc2 = nn.Linear(512, 256)  # Further processing to get the final feature vector.

    def forward_once(self, image, weather):
        # Process an image through the image network.
        image_output = self.img_net(image)
        # Transform the output to have the desired dimension.
        image_output = self.fc_img_1(image_output)

        # Process weather data through the RNN.
        weather_output, _ = self.rnn(weather)
        # Only use the final output of the RNN.
        weather_output = weather_output[:, -1, :]

        # Combine the outputs from the image and weather networks.
        combined = torch.cat((image_output, weather_output), dim=1)

        # Pass the combined vector through fully connected layers.
        combined = torch.relu(self.fc1(combined))
        combined = self.fc2(combined)

        return combined

    def forward(self, input):
        # Process inputs through the network. Input should be a tuple of (image, weather).
        output = self.forward_once(*input)
        return output


# Normal compared network section
class ComparedNetwork(nn.Module):
    '''
    This architecture employs a network composed of several simple fully connected layers to evaluate the relationship
    between two sets of input features, determining whether they are less than, equal to, or greater than each other.
    The input features are derived from the output of above FeatureExtractNetwork.
    '''
    def __init__(self):
        # Define a sequence of fully connected layers
        super(ComparedNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # Matching the input dimension to the feature dimension
            nn.ReLU(),  # Activation function
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, output1, output2):
        # Concatenate the two outputs directly
        combined_output = torch.cat((output1, output2), dim=1)
        # Calculate class probabilities through the network
        class_probabilities = self.fc(combined_output)
        return class_probabilities


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size  # Size of each input token's embedding
        self.heads = heads  # Number of attention heads
        self.head_dim = embed_size // heads # Dimension of each attention head

        # Ensure the embedding size is divisible by the number of heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        # Linear layers for transforming inputs
        self.values = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.keys = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.queries = nn.Linear(embed_size, self.head_dim * heads, bias=False)

        # Output linear layer
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]

        # Transform and then reshape for multi-head attention
        '''
        Adding an extra dimension is crucial for adapting 2D input data [batch size, vector] to the multi-head 
        self-attention mechanism. This extra dimension allows the model to handle multiple "heads," facilitating 
        parallel processing and enabling the model to capture diverse representations of the data. It transforms the 
        input into a format [batch size, sequence length, heads, head dimension] suitable for multi-head attention, 
        enhancing the model's ability to learn complex data patterns.
        '''
        # Apply the linear transformation first, then reshape for multi-head attention
        values = self.values(values).reshape(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(N, -1, self.heads, self.head_dim)
        queries = self.queries(queries).reshape(N, -1, self.heads, self.head_dim)

        # Compute the attention (energy) scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Apply attention to the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, -1, self.heads * self.head_dim)

        # Final linear transformation
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention mechanism
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        # Layer normalization and dropout for stability and regularization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Feedforward network within transformer block
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        # Apply attention
        attention = self.attention(value, key, query)
        # Apply dropout, add the input (residual connection), and normalize
        x = self.dropout(self.norm1(attention + query))
        # Pass through feedforward network, dropout, add residual, and normalize
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class ComparedNetwork_Transformer(nn.Module):
    def __init__(self, embed_size=512, num_layers=1, heads=4, dropout=0.1, output_size=2):
        super(ComparedNetwork_Transformer, self).__init__()
        self.embed_size = embed_size  # Embedding size for each token
        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout) for _ in range(num_layers)]
        )

        # Fully connected layers for final classification
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 256), # Matching the input dimension to the feature dimension
            nn.ReLU(),  # Activation function
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, output_size)
        )

    def forward(self, output1, output2):
        out = torch.cat((output1, output2), dim=1)  # Concatenating on the feature dimension
        for layer in self.layers:
            out = layer(out, out, out)
        # print("Shape before fc:", out.shape)

        # Reduce across sequence length to single vector per sample (more information in MultiHeadSelfAttention class),
        # mean or max strategy, better result in max
        # out = torch.mean(out, dim=1)
        out, _ = torch.max(out, dim=1)

        # Final classification of final layer
        out = self.fc(out)

        return out