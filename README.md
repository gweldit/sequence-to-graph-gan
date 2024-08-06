# sequence-to-graph-gan

# Instructions to run the code:

A`README.md` file with instructions on how to clone a Git repo and run the code:

```markdown
# A Sequence-to-Graph-GAN

This is a Python implementation of a Sequence-to-Graph Generative Adversarial Network (SG-GAN). It implements both WGAN and WGAN-GP to generate sequences and the sequences are encoded to graph for training and evaluating GNN-based models.
```

### Prerequisites

you need to install the software and how to install them.

- Git

### Create Virtual Environment

1. **Create Python Virtual Environment**

```bash
python3 -m venv sggan-venv
```

2. **Activate the Virtual Environment**
   To activate the virtual environment:

   ```bash
   source sggan-venv/bin/activate
   ```

3. **Deactivate the Virtual Environment**
   Whenever you want to deactivate your virtual environment, use the command:
   ```bash
   deactivate
   ```

### Installing

Once you activate directory, you can clone this ropo as follows.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/gweldit/sequence-to-graph-gan.git
   ```

2. **Navigate to the project directory**

   ```bash
   cd sequence-to-graph-gan
   ```

3. **Install dependencies**

   ```bash
    pip3 install -r requirements.txt
   ```

4. **Instructions to Run the project**

- 1. Using the terminal, run the graph_encoder.py to read all the files from the ADFA directory. This will create preprocessed sequences, split them into training and testing, and save them in JSON files in the default (data) folder. Also, the graph_encoder will use the training dataset (sequences) to generate training and testing graph data and save them in train_graph_data.pkl and test_graph_data.pkl file. The dataset folder new to be specified.

  ```bash
  python3 graph_encoder.py --dataset_folder=".../path/ADFA"
  ```

- 2.  To train the GNN model, you can run using the following command in the terminal. This will only train the GNN model using original training data and validate it on the test set:

  ```bash
      python3 train_graph.py
  ```

- 3. To train the generative model (WGAN-GP or WGAN) using the terminal:

  ```bash
      python3 wgan_gp.py --vocab_size=175 --seq_len=512 --batch_size=64 --lr=0.0002 --lambda_gp=10.0 --tau=0.5 --epochs=500
  ```

- 4. To generate fake data (of minority class) using either generative model:
     ```bash
         python3 generate_fake_data.py --n_samples=200
     ```

- 5. To retrain the GNN model using augmeted (with fake added) data:
  ```bash
      python3 train_graph.py --use_fake_data
  ```

## Credit

Some of the codes in this repo are borrowed from this [repository](https://github.com/Willtl/syscall-trace-classification.git).
