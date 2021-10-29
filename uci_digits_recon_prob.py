# uci_digits_recon_prob.py

# Anomaly detection using VAE reconstruction probability
# not same as simple VAE reconstuction Error technique

# PyTorch 1.8.1-CPU Anaconda3-2020.02  (Python 3.7.6)
# Windows 10 

import numpy as np
import scipy.stats as sps
import torch as T
import matplotlib.pyplot as plt

device = T.device("cpu") 

# -----------------------------------------------------------

class UCI_Digits_Dataset(T.utils.data.Dataset):
  # 8,12,0,16, . . 15,7
  # 64 pixel values [0-16], digit [0-9]

  def __init__(self, src_file, n_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=n_rows,
      usecols=range(0,65), delimiter=",", comments="#",
      dtype=np.float32)
    self.xy_data = T.tensor(all_xy, dtype=T.float32).to(device) 
    self.xy_data[:, 0:64] /= 16.0   # normalize 64 pixels
    self.xy_data[:, 64] /= 9.0      # normalize digit/label

  def __len__(self):
    return len(self.xy_data)

  def __getitem__(self, idx):
    xy = self.xy_data[idx]
    return xy                       # includes the label

# -----------------------------------------------------------

def display_digit(ds, idx, save=False):
  # ds is a PyTorch Dataset
  line = ds[idx]  # tensor
  pixels = np.array(line[0:64])  # numpy row of pixels
  label = np.int(line[64] * 9.0)  # denormalize; like '5'
  print("\ndigit = ", str(label), "\n")

  pixels = pixels.reshape((8,8))
  for i in range(8):
    for j in range(8):
      pxl = pixels[i,j]  # or [i][j] syntax
      pxl = np.int(pxl * 16.0)  # denormalize
      print("%.2X" % pxl, end="")
      print(" ", end="")
    print("")

  plt.imshow(pixels, cmap=plt.get_cmap('gray_r'))
  if save == True:
    plt.savefig(".\\idx_" + str(idx) + "_digit_" + \
    str(label) + ".jpg", bbox_inches='tight')
  plt.show() 
  plt.close() 

# -----------------------------------------------------------

def display_digits(ds, idxs, save=False):
  # idxs is a list of indices
  for idx in idxs:
    display_digit(ds, idx, save)

# -----------------------------------------------------------

class VAE(T.nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    # 65-32-[8,8]-8=32-[65,65]-65

    self.input_dim = 65
    self.latent_dim = 8

    self.fc1 = T.nn.Linear(65, 32)
    self.fc2a = T.nn.Linear(32, 8)  # u1
    self.fc2b = T.nn.Linear(32, 8)  # logvar1

    self.fc3 = T.nn.Linear(8, 32)
    self.fc4a = T.nn.Linear(32, 65) # u2
    self.fc4b = T.nn.Linear(32, 65) # logvar2

    # TODO: explicit weight initialization

  # encode: input X to (u, log-var)
  # 65-32-[8,8]
  def encode(self, x):
    z = T.sigmoid(self.fc1(x))   # 65-32
    u1 = self.fc2a(z)         # 32-8
    lv1 = self.fc2b(z)         # 32-8
    return (u1, lv1)  # (mean, log-var)

  # decode: (noise) to generated X
  # 8-32-[65,65]-65
  def decode(self, z): 
    z = T.sigmoid(self.fc3(z))   # 8-32
    u2 = self.fc4a(z)         # 32-65
    lv2 = self.fc4b(z)         # 32-65

    stdev = T.exp(0.5 * lv2)  # T.sqrt(T.exp(v2))
    noise = T.randn_like(stdev)
    z = u2 + (noise * stdev)  # [65,65]-65

    xx = T.sigmoid(z)     # in (0.0, 1.0)
    return (xx, u2, lv2)   # reconstructed x

  # forward: encode+combine+decode
  # 65-32-[8,8]-8-32-[65,65]-65
  def forward(self, x):
    # x values all in (0.0, 1.0)
    (u1,lv1) = self.encode(x)   # [8,8]

    stdev = T.exp(0.5 * lv1)
    noise = T.randn_like(stdev)
    z = u1 + (noise * stdev)   # [8,8]-8 

    (xx, u2, lv2) = self.decode(z)  # all [65]
    return (xx, u2, lv2)

# -----------------------------------------------------------

def cus_loss_func(recon_x, x, u, logvar, beta=1.0):
  # aka ELBO
  # https://arxiv.org/abs/1312.6114
  # bce = T.nn.functional.binary_cross_entropy(recon_x, x, \
  #   reduction="sum")
  mse = T.nn.functional.mse_loss(recon_x, x, reduction="mean")
  kld = -0.5 * T.sum(1 + logvar - u.pow(2) - logvar.exp())
  return mse + (beta * kld)  # beta weights KLD component

# -----------------------------------------------------------

def train(vae, ds, bs, me, le, lr, beta):
  # model, dataset, batch_size, max_epochs,
  # log_every, learn_rate, loss KLD weight
  # assumes vae.train() has been set

  data_ldr = T.utils.data.DataLoader(ds, batch_size=bs,
    shuffle=True)
  opt = T.optim.SGD(vae.parameters(), lr=lr)
  print("\nStarting training")
  for epoch in range(0, me):
    epoch_loss = 0.0
    for (bat_idx, batch) in enumerate(data_ldr):
      opt.zero_grad()
      (recon_x, u, logvar) = vae(batch)
      loss_val = cus_loss_func(recon_x, batch, u, logvar, beta)
      loss_val.backward()
      epoch_loss += loss_val.item()  # average per batch
      opt.step()
    if epoch % le == 0:
      print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
  print("Done ")

# -----------------------------------------------------------

def recon_prob(vae, xi, n_samples=10):
  # xi is one Tensor item, not a batch
  vae.eval()
  
  with T.no_grad():
    (u1, lv1) = vae.encode(xi)
  u1 = u1.numpy()               # mean
  v1 = np.exp(lv1.numpy())      # variance

  # loop L times  (L = n_samples)
  #   make noise from N(u1,v1)
  #   (u2,v2) = decode(noise)
  #   p = prob xi came from N(u2,v2)
  #   sum += p
  # return sum / L
  # possible but not useful to use set approach

  sum_p = 0.0
  for _ in range(n_samples):
    sample = sps.multivariate_normal.rvs(u1, \
      np.diag(v1), size=1)   # random Gaussian sample
    sample = T.tensor(sample, \
      dtype=T.float32).to(device)  # to tensor
    with T.no_grad():
      (xx, u2, lv2) = vae.decode(sample)

    u2 = u2.numpy()
    # v2 = np.exp(lv2)
    v2 = np.ones(vae.input_dim, dtype=np.float32)  # estimated
    covar_mat = np.diag(v2)
    p = sps.multivariate_normal.logpdf( xi, u2, covar_mat)
    sum_p += p

  recon_p = sum_p / n_samples
  return recon_p

# -----------------------------------------------------------

def make_prob_list(vae, ds):
  vae.eval()
  result_lst = []
  for i in range(len(ds)):
    xi = ds[i]
    lp = recon_prob(vae, xi, n_samples=10)
    result_lst.append((i,lp))
  return result_lst 

# -----------------------------------------------------------

def main():
  # 0. preparation
  print("\nBegin UCI Digits VAE reconstuction prob demo ")
  T.manual_seed(1)
  np.random.seed(1)

  # 1. create data objects
  print("Creating UCI Digits_100 Dataset ")
  data_file = ".\\Data\\uci_digits_train_100.txt"
  data_ds = UCI_Digits_Dataset(data_file) 

  # 2. create variational autoencoder model
  print("\nCreating 65-32-[8,8]-8=32-[65,65]-65 VAE network ")
  vae = VAE().to(device)
  vae.train()   # set mode

# -----------------------------------------------------------

  # 3. train VAE reconstruction probability model
  bat_size = 10
  max_epochs = 10
  log_interval = 2
  lrn_rate = 0.001
  beta = 1.0  # KLD weighting in custom_loss

  print("\nbat_size = %3d " % bat_size)
  print("loss = custom MSE plus (beta * KLD) ")
  print("loss beta = %0.2f " % beta)
  print("optimizer = SGD")
  print("max_epochs = %3d " % max_epochs)
  print("lrn_rate = %0.3f " % lrn_rate)

  train(vae, data_ds, bat_size, max_epochs, 
    log_interval, lrn_rate, beta)

# -----------------------------------------------------------

  # 4. make a fake image as a sanity check
  # vae.eval()
  # rinpt = T.randn(1, vae.latent_dim).to(device) 
  # with T.no_grad():
  #   (fi, _, _) = vae.decode(rinpt)
  # print("\nFake generated image (normed): ")
  # print(fi)  # 65 values in 0.0, 1.0

  # 5. TODO: save final model

# -----------------------------------------------------------

  # 6. reconstruction prob for single synthetic item
  # vae.eval()  # set mode
  # x = T.ones(65, dtype=T.float32)
  # x /= 4.0
  # print("\nSetting synthetic image all 65 vals == 0.25 ")
  # lp = recon_prob(vae, x, n_samples=10)
  # print("Reconstruction log(p) = %0.6f " % lp)
  # input()

# -----------------------------------------------------------

  # 7. compute and store reconstruction probabilities
  print("\nComputing reconstruction probabilities ")
  prob_lst = make_prob_list(vae, data_ds)
  prob_lst.sort(key=lambda x: x[1], \
    reverse=False)  # low prob (anomalous) to high prob

# -----------------------------------------------------------

  # 8. show anomalous items
  n = 5

  # small log(p) items
  print("\nTop " + str(n) + " anaomalies -- small log(p): ")
  for i in range(n):
    print(prob_lst[i])

  # large log(p) could be "too good to be true"
  print("\nTop " + str(n) + " suspicious -- large log(p): ")
  for i in range(99, 99-n, -1):  # there are 100 items
    print(prob_lst[i])

  print("\nMost anomalous -- small log(p) -- digit: ")
  idx = prob_lst[0][0]  # first item, index
  display_digit(data_ds, idx)

  print("\nEnd UCI Digits VAE recon prob demo \n")

# -----------------------------------------------------------

if __name__ == "__main__":
  main()
