import argparse
import numpy as np
import torch

from stimulus import StimGenerator
from models import STPNet, OptimizedRNN, STPRNN
from utilities import train


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Models of change detection')
    parser.add_argument('--image-set', type=str, default='A', metavar='I',
                        help='image set to train on: A, B, C, D (default: A)')
    parser.add_argument('--model', type=str, default='STPNet', metavar='M',
                        help='model to train: STPNet, RNN, or STPRNN (default: STPNet)')
    parser.add_argument('--noise-std', type=float, default=0.0, metavar='N',
                        help='standard deviation of noise (default: 0.0)')
    parser.add_argument('--hidden-dim', type=int, default=16, metavar='N',
                        help='hidden dimension of model (default: 16)')
    parser.add_argument('--l2-penalty', type=float, default=0.0, metavar='L2',
                        help='L2 penalty on hidden activations (default: 0.0)')
    parser.add_argument('--pos-weight', type=float, default=1.0, metavar='W',
                        help='weight on positive examples (default: 1.0)')
    parser.add_argument('--seq-length', type=int, default=50000, metavar='N',
                        help='length of each trial (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='number of train trial batches (default: 128)')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='epoch train criterion (default: 5000)')
    parser.add_argument('--dprime', type=float, default=2.0, metavar='N',
                        help='dprime train criterion (default: 2.0)')
    parser.add_argument('--patience', type=int, default=1, metavar='N',
                        help='number of epochs to wait above baseline (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create train stimulus generator
    train_generator = StimGenerator(image_set=args.image_set, seed=args.seed,
                                    batch_size=args.batch_size, seq_length=args.seq_length)

    # Get input dimension of feature vector
    input_dim = len(train_generator.feature_dict[0])

    # Create model
    if args.model == 'STPNet':
        model = STPNet(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       noise_std=args.noise_std).to(device)
    if args.model == 'STPRNN':
        model = STPRNN(input_dim=input_dim,
                       hidden_dim=args.hidden_dim,
                       noise_std=args.noise_std).to(device)
    elif args.model == 'RNN':
        model = OptimizedRNN(input_dim=input_dim,
                             hidden_dim=args.hidden_dim,
                             noise_std=args.noise_std).to(device)
    else:
        raise ValueError("Model not found")

    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor([args.pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize tracking variables
    loss_list = []
    dprime = 0
    dprime_list = []
    wait = 0

    for epoch in range(1, args.epochs + 1):
        # Train model
        loss, dprime = train(args, device, train_generator,
                             model, criterion, optimizer)

        loss_list.append(loss)
        dprime_list.append(dprime)

        if epoch % args.log_interval == 0:
            # Print current progress
            print("Epoch: {}  loss: {:.4f}  dprime: {:.2f}".format(
                epoch, loss, dprime))

        if dprime < args.dprime:
            # Reset wait count
            wait = 0
        else:
            # Increase wait count
            wait += 1
            # Stop training after wait exceeds patience
            if wait >= args.patience:
                break

    # Save trained model
    save_path = './PARAM/' + args.model + \
        '/model_train_seed_'+str(args.seed)+'.pt'
    torch.save({'epoch': epoch,
                'loss': loss_list,
                'dprime': dprime_list,
                'state_dict': model.state_dict()}, save_path)


if __name__ == '__main__':
    main()
