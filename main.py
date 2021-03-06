import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as tF
from torch.nn import init

def rgb2ycbcr(im):
    im = np.array(im)
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return tF.to_pil_image(np.uint8(ycbcr)[:, :, [0]])

class RGB2Y(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """ 

        return rgb2ycbcr(img)

parser = argparse.ArgumentParser(description='Neural Processes (NP) for MNIST image completion')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset for training')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--r_dim', type=int, default=128, metavar='N',
                    help='dimension of r, the hidden representation of the context points')
parser.add_argument('--z_dim', type=int, default=128, metavar='N',
                    help='dimension of z, the global latent variable')
parser.add_argument('--hidden_dim', type=int, default=128, metavar='N',
                    help='dimension of z, the global latent variable')
parser.add_argument('--att_type', type=str, default='uniform',
                    help='attention type')
parser.add_argument('--rep_type', type=str, default='identity',
                    help='representation type')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.dataset == 'mnist':
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],
                                     std=[1])])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    channel_dim, data_dim = 28, 784
    c = 1
elif args.dataset == 'cifar10':
    # transform_ = transforms.Compose([RGB2Y(), transforms.ToTensor(), transforms.Normalize(mean=[0.5],
    #                                  std=[1])])
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=True, download=True,
                       transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=False, transform=transform_),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    channel_dim = 32
    c = 3
    data_dim = channel_dim * channel_dim * 1
    

 

def get_context_idx(N):
    # generate the indeces of the N context points in a flattened image
    idx = random.sample(range(0, data_dim), N)
    idx = torch.tensor(idx, device=device)
    return idx


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, batch_size):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x


class NP(nn.Module):
    def __init__(self, args):
        super(NP, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.hidden_dim = args.hidden_dim

        self.h_1 = nn.Linear(c + 2, self.hidden_dim)
        self.h_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_3 = nn.Linear(self.hidden_dim, self.r_dim)

        self.r_dense = nn.Linear(self.r_dim, self.r_dim)
        self.r_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)

        self.h_4 = nn.Linear(c + 2, self.hidden_dim)
        self.h_5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_6 = nn.Linear(self.hidden_dim, self.r_dim)
        self.s = nn.Linear(self.r_dim, self.z_dim)

        if args.rep_type == 'mlp':
            self.att_rep_k = nn.Linear(self.z_dim, self.z_dim)
            self.att_rep_k = nn.Linear(self.z_dim, self.z_dim)
        
        self.num_heads = 4

        if args.att_type == 'multihead':
            d_k, d_v = 2, self.r_dim
            self.heads = []
            head_size = int(self.r_dim / self.num_heads)

            for i in range(self.num_heads):
                self.heads.append(nn.Conv1d(d_k, d_k, 1))
                self.heads[-1].weight.data.normal_(0, d_k**(-0.5))
                self.heads.append(nn.Conv1d(d_k, d_k, 1))
                self.heads[-1].weight.data.normal_(0, d_k**(-0.5))
                self.heads.append(nn.Conv1d(d_v, head_size, 1))
                self.heads[-1].weight.data.normal_(0, d_k**(-0.5))
                self.heads.append(nn.Conv1d(head_size, self.r_dim, 1))
                self.heads[-1].weight.data.normal_(0, d_v**(-0.5))

            self.heads = nn.Sequential(*(self.heads))

        self.g_1 = nn.Linear(self.z_dim * 2 + 2, self.hidden_dim)
        self.g_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.g_5 = nn.Linear(self.hidden_dim, c)
        self.g_mean = nn.Linear(self.hidden_dim, c)
        self.g_var = nn.Linear(self.hidden_dim, c)

        self.attention_type = args.att_type
        

    def h(self, x_y):
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(x_y))
        x_y = F.relu(self.h_3(x_y))
        return x_y

    def h_s(self, x_y):
        x_y = F.relu(self.h_4(x_y))
        x_y = F.relu(self.h_5(x_y))
        x_y = F.relu(self.h_6(x_y))
        return x_y

    def dot_product_attention(self, q, k, v, normalise):
        """Computes dot product attention.

        Args:
          q: queries. tensor of  shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          normalise: Boolean that determines whether weights sum to 1.
        
        Returns:
          tensor of shape [B,m,d_v].
        """
        # print (q.shape, k.shape, v.shape)
        d_k = q.shape[-1]
        scale = np.sqrt(1.0 * d_k)
        unnorm_weights = torch.einsum('bjk,bik->bij', (k, q)) / scale  # [B,m,n]
        if normalise:
            weight_fn = torch.nn.functional.softmax
        else:
            weight_fn = torch.sigmoid
    
        weights = weight_fn(unnorm_weights, dim=-1)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', (weights, v))  # [B,m,d_v]
        return rep

    def multihead_attention(self, q, k, v, num_heads=8):
        """Computes multi-head attention.
  
        Args:
          q: queries. tensor of  shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          num_heads: number of heads. Should divide d_v.
          
        Returns:
          tensor of shape [B,m,d_v].
        """
        d_k = q.shape[-1]
        d_v = v.shape[-1]
        head_size = d_v / num_heads
        
        rep = 0.0
        for h in range(num_heads):
            o = self.dot_product_attention(
                self.heads[h * 4](q.permute(0,2,1)).permute(0,2,1),
                self.heads[h * 4 + 1](k.permute(0,2,1)).permute(0,2,1),
                self.heads[h * 4 + 2](v.permute(0,2,1)).permute(0,2,1),
                normalise=True)
            
            rep += self.heads[h * 4 + 3](o.permute(0,2,1)).permute(0,2,1)
        return rep

    def aggregate(self, content_x, target_x, r):
        if self.attention_type == 'uniform':
            return torch.mean(r, dim=1).unsqueeze(1).expand(-1, data_dim, -1)
        elif self.attention_type == 'dot':
            return self.dot_product_attention(target_x, content_x, r, True)
        elif self.attention_type == 'multihead':
            return self.multihead_attention(target_x, content_x, r, self.num_heads)


    def reparameterise(self, z):
        mu, var = z

        m = torch.distributions.normal.Normal(mu, var)
        z_sample = m.rsample()
        
        z_sample = z_sample.unsqueeze(1).expand(-1, data_dim, -1)
        
        return z_sample

    def g(self, s_sample, z_sample, x_target):
        z_x = torch.cat([s_sample, z_sample, x_target], dim=2)
        
        input = F.relu(self.g_1(z_x))
        input = F.relu(self.g_2(input))
        input = F.relu(self.g_3(input))
        input = F.relu(self.g_4(input))
        
        y_mean = self.g_mean(input)
        y_var = self.g_var(input)
        sigma = 0.1 + 0.9 * F.softplus(y_var)
        
        y_dis = torch.distributions.normal.Normal(y_mean, sigma)
        
        y_hat = y_mean 
        return (y_hat, y_dis)

    def xy_to_z_params(self, x, y):
        x_y = torch.cat([x, y], dim=2)
        r_i = self.h(x_y)

        r = torch.mean(r_i, dim=1)

        r = F.relu(self.r_dense(r))

        mu = self.r_to_z_mean(r)
        logvar = self.r_to_z_logvar(r)
        var = (0.1 + 0.9 * torch.sigmoid(logvar)) * 0.1
        return mu, var

    def xy_to_s_params(self, x, y, x_target):
        x_y = torch.cat([x, y], dim=2)
        r_i = self.h_s(x_y)
        r = self.aggregate(x, x_target, r_i)
        
        return self.s(r)

    def forward(self, x_context, y_context, x_all=None, y_all=None):
        x_target = x_grid.expand(y_context.shape[0], -1, -1)

        z_context = self.xy_to_z_params(x_context, y_context)  # (mu, logvar) of z
        s_contenxt = self.xy_to_s_params(x_context, y_context, x_target)

        if self.training:  # loss function will try to keep z_context close to z_all
            z_all = self.xy_to_z_params(x_all, y_all)
            # s_all = self.xy_to_s_params(x_all, y_all)
            s_all = s_contenxt
        else:  # at test time we don't have the image so we use only the context
            z_all = z_context
            s_all = s_contenxt

        z_sample = self.reparameterise(z_all)
        s_sample = s_all

        # reconstruct the whole image including the provided context points
        
        y_hat = self.g(s_sample, z_sample, x_target)

        return y_hat, z_all, z_context


def kl_div_gaussians(mu_q, var_q, mu_p, var_p):
    # var_p = torch.exp(logvar_p)
    logvar_p, logvar_q = torch.log(var_p), torch.log(var_q)
    kl_div = (var_q + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum() / kl_div.shape[0]
    return kl_div


def np_loss(y_hat, y, z_all, z_context):
    y_hat, y_dis = y_hat
    log_p = y_dis.log_prob(y).sum(dim = -1).sum(dim = -1)
    BCE = - log_p.sum() / log_p.shape[0]
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1]) 
    return BCE + KLD


model = NP(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
x_grid = generate_grid(channel_dim, channel_dim)
os.makedirs("results/", exist_ok=True)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (y_all, _) in enumerate(train_loader):
        batch_size = y_all.shape[0]
        if args.dataset == 'mnist':
            y_all = y_all.to(device).view(batch_size, -1, 1)
        elif args.dataset == 'cifar10':
            y_all = y_all.permute(0,2,3,1).to(device).view(batch_size, -1, c)
        
        N = random.randint(1, data_dim)  # number of context points
        context_idx = get_context_idx(N)
        
        x_context = idx_to_x(context_idx, batch_size)
        y_context = idx_to_y(context_idx, y_all)
        
        x_all = x_grid.expand(batch_size, -1, -1)
        
        optimizer.zero_grad()
        y_hat, z_all, z_context = model(x_context, y_context, x_all, y_all)

        loss = np_loss(y_hat, y_all, z_all, z_context)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(y_all), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(y_all)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (y_all, _) in enumerate(test_loader):
        
            batch_size = y_all.shape[0]
            if args.dataset == 'mnist':
                y_all = y_all.to(device).view(batch_size, -1, 1)
            elif args.dataset == 'cifar10':
                y_all = y_all.to(device).permute(0,2,3,1).view(batch_size, -1, c)
            
            N = 300
            context_idx = get_context_idx(N)
            x_context = idx_to_x(context_idx, batch_size)
            y_context = idx_to_y(context_idx, y_all)

            y_hat, z_all, z_context = model(x_context, y_context)
            test_loss += np_loss(y_hat, y_all, z_all, z_context).item()

            if i == 0:  # save PNG of reconstructed examples
                plot_Ns = [int(data_dim * 0.1), int(data_dim * 0.3), int(data_dim * 0.5), data_dim]
                num_examples = min(batch_size, 16)
                for N in plot_Ns:
                    recons = []
                    context_idx = get_context_idx(N)
                    x_context = idx_to_x(context_idx, batch_size)
                    y_context = idx_to_y(context_idx, y_all)
                    for d in range(5):
                        y_hat, _, _ = model(x_context, y_context)
                        y_hat = y_hat[0]
                        recons.append(y_hat[:num_examples])
                        
                    recons = torch.cat(recons).permute(0,2,1).view(-1, c, channel_dim, channel_dim).expand(-1, 3, -1, -1)
                    background = torch.tensor([0., 0., 1.], device=device)
                    background = background.view(1, -1, 1).expand(num_examples, 3, data_dim).contiguous()
                    
                    if args.dataset == 'mnist':
                        context_pixels = y_all[:num_examples].view(num_examples, 1, -1)[:, :, context_idx]
                        context_pixels = context_pixels.expand(num_examples, 3, -1)
                        background[:, :, context_idx] = context_pixels
                        comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                                recons]) + 0.5
                    elif args.dataset == 'cifar10':
                        context_pixels = y_all[:num_examples].permute(0,2,1)[:, :, context_idx]
                        context_pixels = context_pixels.expand(num_examples, 3, -1)
                        background[:, :, context_idx] = context_pixels
                        comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                                recons]) + 0.5
                    save_image(comparison.cpu(),
                               'results/%s_ep_' % (args.dataset + '_' + args.att_type) + str(epoch) +
                               '_cps_' + str(N) + '.png', nrow=num_examples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
