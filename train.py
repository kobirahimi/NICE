"""Training procedure for NICE.
"""
import argparse
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, device):
    flow.train()  # set to training mode
    total_loss = 0
    for num_batch, inputs in enumerate(trainloader, 1):
        inputs, _ = inputs
        inputs = inputs.view(inputs.shape[0], inputs.shape[1]*inputs.shape[2]*inputs.shape[3])
        noise = torch.distributions.Uniform(0., 1.).sample(inputs.size())
        inputs = (inputs * 255. + noise) / 256.
        inputs.to(device)
        optimizer.zero_grad()
        loss = -flow(inputs).mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    return total_loss / num_batch


def test(flow, testloader, filename, epoch, sample_shape, device):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        a, b = samples.min(), samples.max()
        samples = (samples-a)/(b-a+1e-10)
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + '_epoch_%d.png' % epoch)

        total_loss = 0
        for num_batch, inputs in enumerate(testloader, 1):
            inputs, _ = inputs
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
            noise = torch.distributions.Uniform(0., 1.).sample(inputs.size())
            inputs = (inputs * 255. + noise) / 256.
            inputs.to(device)
            loss = -flow(inputs).mean()
            total_loss += float(loss)
    return total_loss / num_batch


def main(args):
    device = 'cpu'
    if args.dataset == 'mnist':
        (full_dim, mid_dim, hidden) = (1 * 28 * 28, args.mid_dim, args.hidden)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='~/torch/data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=1)

    elif args.dataset == 'fashion-mnist':
        (full_dim, mid_dim, hidden) = (1 * 28 * 28, args.mid_dim, args.hidden)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/FashionMNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=1)

    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%d_' % args.coupling \
             + 'coupling_type%s_' % args.coupling_type \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + '.pt'

    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                coupling_type=args.coupling_type,
                in_out_dim=full_dim,
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        train_loss = train(flow, trainloader, optimizer, device)
        train_losses.append(train_loss)
        filename = f"samples of {args.dataset}"
        sample_shape = [1, 28, 28]
        test_loss = test(flow, testloader, filename, epoch+1, sample_shape, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1} finished:  train loss: {train_loss}, test loss: {test_loss} ")

        if epoch % 10 == 0:  # Save model every 10 epochs
            torch.save(flow.state_dict(), "./models/" + model_save_filename)

    with torch.no_grad():
        fig, ax = plt.subplots()
        ax.plot(train_losses)
        ax.plot(test_losses)
        ax.set_title("Train and Test Log Likelihood Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["train loss", "test loss"])
        plt.savefig("./loss/" + f"{args.dataset}_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
