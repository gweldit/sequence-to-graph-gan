# load pytorch model from saved models 

import argparse
import os
import pickle
import torch
from helper_tools import get_device
from wgan_gp import Generator
from wgan import Generator as LSTMGenerator
from graph_encoder import GraphEncoder

def load_model(model_name, args):
    """
    Load the model from the saved model directory   

    Args:   
        model_name (str): The name of the model to load.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    
    # check if file exists 
    file_path =f"saved_models/{model_name}"
    
    if os.path.exists(file_path):
        if file_path.endswith(".pth"):
            model = Generator(vocab_size=args.vocab_size)
            model.load_state_dict(torch.load(file_path))
            return model
        elif file_path.endswith(".pt"):
            model = LSTMGenerator(args.vocab_size, args.num_classes, args.num_layers, softmax=args.softmax)
            model.load_state_dict(torch.load(file_path))
            return model
    else:
        raise FileNotFoundError


def gen_fake_data(generator_name, labels, args, output_file_name:str="data/fake_graph_data.pkl", device=torch.device('cpu')):
    """
    Generate fake data using the provided model.

    Args:
        generator (torch.nn.Module): The model to use for generating fake data.
        labels (int): The labels to generate.
        tau (float): The temperature to generate tokens 
        hard (bool): to get perfect one-hot encoding or floating points like one-hot.
        output_file_name (str): file name to save generated data.

    Returns:
        fake_data (torch.Tensor): The generated fake data.
    """

    
    encoder = GraphEncoder()
    loaded_vocabs = encoder.load_vocabulary()
    vocabs = {i:i for i in range(len(loaded_vocabs))}
    vocab_size = len(vocabs)
    device = labels.device
    generator = load_model(generator_name, args).to(device)
    generator.eval() 

    with torch.no_grad():
        noise = torch.randn((labels.size(0), generator.latent_dim)).to(device)
        # labels = labels.to(device)
        gen_data = generator(noise, labels, args.tau, args.hard).argmax(-1)
        print("sample generated data: ", gen_data[0])

        # print("min and max values of gen data = ", gen_data.min(), gen_data.max())

    # save generated data to a pickle file in data/fake_graph_data.pkl
    # with open("data/fake_graph_data.pkl", "wb") as f:
        # save generated data and labels
        fake_data = gen_data.cpu().detach().tolist()
        fake_labels = labels.cpu().detach().tolist()
        # set syscall vocabs to the encoder object, making it compatible with the original data encoding.
        setattr(encoder, "syscall_vocabs", vocabs)

        gen_graphs, _ = encoder.fetch_graphs(fake_data, fake_labels)

        # save graph data
        encoder.save_graph_data(gen_graphs,output_file_name)


    return gen_data, labels, vocabs


def main():

    # load model
    parser = argparse.ArgumentParser(description='generate fake data based on the model saved at specified epoch')
    parser.add_argument('--model_name', type=str, default='wgan_generator_100', help='name of the wgan generator model at a given epoch')
                        
    # parser.add_argument('--epoch', type=int, default=100, help='epoch value')
    parser.add_argument('--n_samples', type=int, default=100, help='number of fake samples to generate for balancing / augmentation')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature value to control the sampling')
    parser.add_argument('--hard', action="store_true", help='to get hard one-hot or floating point')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--vocab_size', type=int, default=175)

    parser.add_argument('--softmax', type=str, default='gumbel_softmax')



    args = parser.parse_args()

    print("arguments passed are:", args)

    model_name = args.model_name # + "_"+ str(args.epoch)

    # generate noise and fake labels
    
    device = get_device()
    # generate and save graph data
    labels = torch.zeros((args.n_samples,), dtype=torch.long).to(device)
    gen_fake_data(model_name, labels, args)

    # print("max value in generated data: ", gen_data.max())


if __name__ == "__main__":
    main()