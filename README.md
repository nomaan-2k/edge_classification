# Edge Classifier Graph Neural Network

The task is to train an edge classifier graph neural network to classify edges of graphs as true or false. 
Each graph is represented by a PyTorch Geometric (PyG) Data object with the following format:
  
> Data(x=[290, 6], edge_index=[2, 2690], edge_attr=[2690, 4], y=[2690])

- Each node has 6 attributes
- Each edge has 4 attributes
- edge_index stores the indices of the nodes that are connected by each edge.
- Array y stores a binary label (True or False) for each edge describing its class





### To run the model :
- either use the edge_classify.ipynb notebook 
- or run the python script

**NOTE -  UPDATE PATH TO DATASET BEFORE RUNNING JUPYTER NOTEBOOK !!** 


### Achieved Accuracy -
![accuracy](https://user-images.githubusercontent.com/52543604/224617701-5108aae7-736d-44e8-b593-6bf16f117d54.png)
