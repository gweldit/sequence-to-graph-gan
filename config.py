import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=343)
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--seq_len", type=int, default=256, help="length of each sequence dimension")
    
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--hard", type=bool, default=False)
    parser.add_argument("--tau",type=float, default=0.5)


    opt = parser.parse_args()
    print(opt)

    return opt