
''' Edge Classifier GNN Model for binary classification of edges of graphs '''

from pathlib import Path    #for dataset path
from torch.utils.data import Dataset    
import torch                #pytorch
from torch.utils.data import random_split    # to split data 
from torch_geometric.loader import DataLoader    # dataloader
from torch.nn import Linear             # for linear classification 
from torch_geometric.nn import GCNConv    # for Graph Convolutional Networks
from torch_geometric.logging import log    # to log data


# custom dataset class, provided in task
class MyDataset(Dataset):
    def __init__(self, path: Path):
        super().__init__()
        self.graphs = list(path.glob("*.pt"))    

    def __getitem__(self, idx):
        return torch.load(self.graphs[idx])   

    def __len__(self) -> int:
        return len(self.graphs)
    


# load the dataset
# !! UPDATE PATH BEFORE RUNNING SCRIPT !!

dataset = MyDataset(Path("/path/to/my/extracted/data"))



# Split the dataset into a 70/30 ratio for training and testing respectively
train_size = int(0.7 * len(dataset))  
test_size = len(dataset) - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# load mini batched of data
train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)




# Define edge classifier model

class Edge_classifier(torch.nn.Module):


    def __init__(self, input_layer, hidden_layer1, hidden_layer2): 
        super(Edge_classifier, self).__init__()

        # two graph convolutional layers
        self.conv1 = GCNConv(input_layer, hidden_layer1)
        self.conv2 = GCNConv(hidden_layer1, hidden_layer2)

        # linear layer to classify the edges
        self.lin = Linear(2 * hidden_layer2 + 4, 2)         
        
        # (2*hidden_layer2 + 4) due to concatenation of x_src, edge_attr & x_dst into edge_feat


    # forward method to perform forward pass
    def forward(self, x, edge_index, edge_attr):

        R = torch.nn.ReLU()    # activation function

        # convolution layers with ReLU activation
        x = R(self.conv1(x, edge_index))
        x = R(self.conv2(x, edge_index))

        # Concatenate the source node features, destination node features, and edge attributes
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)

        # linear classifier
        # this classifies edges as true or false using edge_feat as input
        edge_pred = self.lin(edge_feat)
        
        # we can use softmax method from torch.nn.functional to get probabilities
        # but as we are using cross entropy loss function, it can handle the predicted output directly

        # return predicted edge labels
        return edge_pred



# Instantiate the Edge Classifier model
model = Edge_classifier(6,8,4)

# cross entropy loss
criterion = torch.nn.CrossEntropyLoss()

# adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



# define a training method

def train():

    # training mode
    model.train()

    total_loss = 0

    # iterate over the training data loader
    for data in train_loader:

        # set gradients to zero
        optimizer.zero_grad()

        # forward pass through the model
        out = model(data.x, data.edge_index, data.edge_attr)

        # calculate loss
        loss = criterion(out, data.y.long())

        # backward pass and update weights and biases
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * data.num_graphs
        
    # return average loss
    return total_loss / len(train_loader.dataset)


# define a testing method

def test(loader):

    # evaluation mode
    model.eval()

    total_correct = 0      # correct predictions
    num_pred = 0           # total predictions made

    # iterate over the data loader
    for data in loader:

        # forward pass
        pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

        # no of correct predictions
        total_correct += (pred == data.y.long()).sum()

        # update total number of predictions
        num_pred += len(pred)
        
    # return the accuracy
    return int(total_correct) / int(num_pred)




# train the model for 200 epochs
for epoch in range(200):

    # trains the model and returns average loss for each epoch
    loss = train()

    # get training and test accuracy at each step
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    # log the data
    log(Epoch=epoch, Loss=loss, Train_Accuracy=train_acc, Test_Accuracy=test_acc)



