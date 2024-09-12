# Mantis: Detection of Zero-Day Malicious Domains Leveraging Low Reputed Hosting Infrastructure

## Overview
Mantis is a tool designed to detect zero-day malicious domains by leveraging low-reputed hosting infrastructures.  It utilizes a hybrid approach, combining network topology with hosting and lexical feature sets to enhance detection capabilities. By leveraging the power of Graph Neural Networks (GNNs), Mantis identifies malicious domains more effectively, offering a robust solution for proactive cybersecurity.

## Sample Data
Sample nodes and edges data can be accessed from the following link:
[Sample Data - Google Drive](https://drive.google.com/drive/folders/1rVDY26SO9xUp4DDZLQ7DuhSJ3B9r0wGQ?usp=sharing)

## Usage

To train a model, follow the steps below:

### 1. Import Required Libraries

```python
import torch
from src import config, sage_experiment, loader
from src.features import feature_labels
from src.score import roc, prec_recall, score
```

### 2. Set Up Your Environment
- Nodes File: The file containing nodes data (fqdn_apex_nodes.csv).
- Edges File: The file containing edges data (fqdn_apex_edges.csv).
- Model File: Define where the trained model will be saved (sage.pkl).

```python
nodes_file = 'fqdn_apex_nodes.csv'
edges_file = 'fqdn_apex_edges.csv'
model_file = 'sage.pkl'
```

### 3. Configure Experiment
You can configure the number of layers and specify the GPU ID and also change other parameters as shown in the example below:

```python
args = config.parse()
args.num_layers = 3
args.dim = 256
args.gpu_id = 5
```

### 4. Load Data and Train Model
Use the loader to load the nodes and edges data.
You can initiate the training process using the SAGE_Experiment class.

```python
data_loader = loader.Loader(nodes_file, edges_file, feature_labels, args)
experiment = sage_experiment.SAGE_Experiment(data_loader.data, args)
model = experiment.train()
torch.save(model.state_dict(), model_file)
```
