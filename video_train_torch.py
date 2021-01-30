import torch
import argparse
import gan
import gan_utils
import data_utils
import glob
import os
import time
import tqdm

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import urllib.request, json
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

print('Pytorch version:', torch.__version__)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"   
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ngpu = 0
start_time = time.time()


def train(args):
    test = args.test
    dname = args.dname
    time_steps = args.time_steps
    batch_size = args.batch_size
    path = args.path
    print(path)
    seed = args.seed
    save_freq = args.save_freq

    with urllib.request.urlopen("https://konstantinklemmer.github.io/data/data_stml/lgcp_64_1000_gneiting.json") as url:
        data = np.array(json.loads(url.read().decode())).astype(np.float32)
    data = torch.tensor(data)
    x_height, x_width, total_time_steps = data.shape

    data_seq = torch.reshape(data, (x_height, x_width, total_time_steps//time_steps, time_steps))
    raw_data = data_seq.permute(0, 2, 1, 3).permute(1, 0, 2, 3).permute(0, 1, 3, 2).permute(0, 2, 1, 3)
    # normalise data between 0 and 1
    data = (raw_data - torch.min(raw_data)) / (torch.max(raw_data) - torch.min(raw_data))

    dataset = data_utils.MyDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size*2, drop_last=True)
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # filter size for (de)convolutional layers
    g_state_size = args.g_state_size
    d_state_size = args.d_state_size
    g_filter_size = args.g_filter_size
    d_filter_size = args.d_filter_size
    reg_penalty = args.reg_penalty
    nlstm = args.n_lstm
    x_width = 64
    x_height = 64
    channels = args.n_channels
    bn = args.batch_norm
    # Number of RNN layers stacked together
    gen_lr = args.lr
    disc_lr = args.lr
    np.random.seed(seed)

    it_counts = 0
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    time_steps = args.time_steps
    # scaling_coef = 1.0

    # Create instances of generator, discriminator_h and
    # discriminator_m CONV VERSION
    z_width = args.z_dims_t
    z_height = args.z_dims_t
    y_dim = args.y_dims

    generator = gan.VideoDCG(batch_size, time_steps, x_h=x_height, x_w=x_width, filter_size=g_filter_size,
                             state_size=g_state_size, bn=bn, output_act='sigmoid').to(device)
    discriminator_h = gan.VideoDCD(batch_size, time_steps, filter_size=d_filter_size,
                                   nchannel=channels, bn=bn).to(device)
    discriminator_m = gan.VideoDCD(batch_size, time_steps, filter_size=d_filter_size,
                                   nchannel=channels, bn=bn).to(device)

    test_ = dname + '-cot'

    saved_file = "{}_{}{}-{}:{}:{}.{}".format(test_,
                                              datetime.now().strftime("%h"),
                                              datetime.now().strftime("%d"),
                                              datetime.now().strftime("%H"),
                                              datetime.now().strftime("%M"),
                                              datetime.now().strftime("%S"),
                                              datetime.now().strftime("%f"))

    model_fn = test_

    log_dir = "./trained/{}/log".format(saved_file)

    # Create directories for storing images later.
    if not os.path.exists("trained/{}/data".format(saved_file)):
        os.makedirs("trained/{}/data".format(saved_file))
    if not os.path.exists("trained/{}/images".format(saved_file)):
        os.makedirs("trained/{}/images".format(saved_file))

    # GAN train notes
    with open("./trained/{}/train_notes.txt".format(saved_file), 'w') as f:
        # Include any experiment notes here:
        f.write("Experiment notes: .... \n\n")
        f.write("MODEL_DATA: {}\nSEQ_LEN: {}\n".format(
            test_,
            time_steps))
        f.write("STATE_SIZE: {}\nLAMBDA: {}\n".format(
            g_state_size,
            reg_penalty))
        f.write("BATCH_SIZE: {}\nCRITIC_ITERS: {}\nGenerator LR: {}\n".format(
            batch_size,
            gen_lr,
            disc_lr))
        f.write("SINKHORN EPS: {}\nSINKHORN L: {}\n\n".format(
            sinkhorn_eps,
            sinkhorn_l))

    writer = SummaryWriter(log_dir)

    beta1 = 0.5
    beta2 = 0.9
    optimizerG = optim.Adam(generator.parameters(), lr=gen_lr, betas=(beta1, beta2))
    optimizerDH = optim.Adam(discriminator_h.parameters(), lr=disc_lr, betas=(beta1, beta2))
    optimizerDM = optim.Adam(discriminator_m.parameters(), lr=disc_lr, betas=(beta1, beta2))

    epochs = 50
    # with tqdm.trange(epochs, ncols=100, unit="epoch") as ep:
    #    for _ in ep:
    #        it = tqdm.tqdm(ncols=100)

    for _ in range(epochs):
        for x in loader:
            it_counts += 1
            # Train D
            z = torch.randn(batch_size, time_steps, z_height * z_width)
            y = torch.randn(batch_size, y_dim)
            z_p = torch.randn(batch_size, time_steps, z_height * z_width)
            y_p = torch.randn(batch_size, y_dim)
            real_data = x[:batch_size, ...]
            real_data_p = x[batch_size:, ...]

            fake_data = generator(z, y)
            fake_data_p = generator(z_p, y_p)

            h_fake = discriminator_h(fake_data)

            m_real = discriminator_m(real_data)
            m_fake = discriminator_m(fake_data)

            h_real_p = discriminator_h(real_data_p)
            h_fake_p = discriminator_h(fake_data_p)

            m_real_p = discriminator_m(real_data_p)

            real_data = real_data.reshape(batch_size, time_steps, -1)
            fake_data = fake_data.reshape(batch_size, time_steps, -1)
            real_data_p = real_data_p.reshape(batch_size, time_steps, -1)
            fake_data_p = fake_data_p.reshape(batch_size, time_steps, -1)

            loss_d = gan_utils.compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                           sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p,
                                                           m_real_p,
                                                           h_real_p, h_fake_p)

            pm1 = gan_utils.scale_invariante_martingale_regularization(m_real, reg_penalty)
            disc_loss = -loss_d + pm1
            # updating Discriminator H
            discriminator_h.zero_grad()
            disc_loss.backward(retain_graph=True)
            optimizerDH.step()

            # updating Discriminator M
            discriminator_m.zero_grad()
            disc_loss.backward()
            optimizerDM.step()

            # Train G
            z = torch.randn(batch_size, time_steps, z_height * z_width)
            y = torch.randn(batch_size, y_dim)
            z_p = torch.randn(batch_size, time_steps, z_height * z_width)
            y_p = torch.randn(batch_size, y_dim)
            real_data = x[:batch_size, ...]
            real_data_p = x[batch_size:, ...]

            fake_data = generator(z, y)
            fake_data_p = generator(z_p, y_p)

            h_fake = discriminator_h(fake_data)

            m_real = discriminator_m(real_data)
            m_fake = discriminator_m(fake_data)

            h_real_p = discriminator_h(real_data_p)
            h_fake_p = discriminator_h(fake_data_p)

            m_real_p = discriminator_m(real_data_p)

            real_data = real_data.reshape(batch_size, time_steps, -1)
            fake_data = fake_data.reshape(batch_size, time_steps, -1)
            real_data_p = real_data_p.reshape(batch_size, time_steps, -1)
            fake_data_p = fake_data_p.reshape(batch_size, time_steps, -1)

            gen_loss = gan_utils.compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                             sinkhorn_eps,
                                                             sinkhorn_l, real_data_p, fake_data_p, m_real_p,
                                                             h_real_p, h_fake_p)
            # updating Generator
            generator.zero_grad()
            gen_loss.backward()
            optimizerG.step()
            # it.set_postfix(loss=float(gen_loss))
            # it.update(1)

            # ...log the running loss
            writer.add_scalar('Sinkhorn training loss', gen_loss, it_counts)
            writer.flush()

            if torch.isinf(gen_loss):
                print('%s Loss exploded!' % test_)
                # Open the existing file with mode a - append
                with open("./trained/{}/train_notes.txt".format(saved_file), 'a') as f:
                    # Include any experiment notes here:
                    f.write("\n Training failed! ")
                break
            else:
                if it_counts % save_freq == 0 or it_counts == 1:
                    z = torch.randn(batch_size, time_steps, z_height * z_width)
                    y = torch.randn(batch_size, y_dim)
                    samples = generator(z, y)
                    # plot first 5 samples within one image
                    samples = samples[:5].permute(0, 2, 1, 3)
                    img = samples.reshape(1, batch_size * x_height, time_steps * x_width)
                    writer.add_image('Generated images', img, global_step=it_counts)
                    # save model to file
                    save_path = "./trained/{}/ckpts".format(saved_file)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(generator, save_path + '/' + 'generator.pt')
                    torch.save(discriminator_h, save_path + '/' + 'discriminatorH.pt')
                    torch.save(discriminator_m, save_path + '/' + 'discriminatorM.pt')
                    print("Saved all models to {}".format(save_path))
            continue
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cot')
    parser.add_argument('-d', '--dname', type=str, default='spatial_data',
                        choices=['spatial_data'])
    parser.add_argument('-t', '--test',  type=str, default='cot', choices=['cot'])
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-gss', '--g_state_size', type=int, default=32)
    parser.add_argument('-gfs', '--g_filter_size', type=int, default=32)
    parser.add_argument('-dss', '--d_state_size', type=int, default=32)
    parser.add_argument('-dfs', '--d_filter_size', type=int, default=32)
    
    # animation data has T=13 and human action data has T=16
    parser.add_argument('-ts', '--time_steps', type=int, default=10)
    parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)
    parser.add_argument('-reg_p', '--reg_penalty', type=float, default=0.01)
    parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
    parser.add_argument('-Dx', '--Dx', type=int, default=1)
    parser.add_argument('-Dz', '--z_dims_t', type=int, default=5)
    parser.add_argument('-Dy', '--y_dims', type=int, default=20)
    parser.add_argument('-g', '--gen', type=str, default="fc",
                        choices=["lstm", "fc"])
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-p', '--path', type=str,
                        default='/home/')
    parser.add_argument('-save', '--save_freq', type=int, default=500)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-bn', '--batch_norm', type=bool, default=True)
    parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)
    
    # animation original data has 4 channels and human action data has 3
    parser.add_argument('-nch', '--n_channels', type=int, default=1)
    parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
    parser.add_argument('-lp', '--projector', type=bool, default=False)

    args = parser.parse_args()
    
    train(args)
