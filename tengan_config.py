import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--vocab_size', type=int, default=200)
    parser.add_argument('--seq_length', type=int, default=290) # 300, 400
    parser.add_argument('--embedding_dim', type=int, default=64) # 512
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_encoders', type=int, default=3) # 4
    parser.add_argument('--dim_feedforward', type=int, default=64) # 12
    parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_wgan', type=bool, default=False)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--g_steps', type=int, default=3) # higher value is better at estimating and reducing variance in gradients

    parser.add_argument('--lr', type=float, default=0.0002)

    parser.add_argument('--train_file_path', type=str, help='The path to the file containing the sequences of training dataset',default="data/train_dataset.json") 


    parser.add_argument('--test_file_path', type=str, help='The path to the file containing the sequences of testing set',default="data/test_dataset.json") 
    parser.add_argument('--pre_training_disc', type=bool, default=False)
    parser.add_argument('--pre_training_gen',type=bool, default=False)
    parser.add_argument('--adversarial_training',type=bool, default=False)

    args = parser.parse_args()

    return args


