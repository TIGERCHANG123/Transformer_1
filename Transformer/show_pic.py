import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from threading import Thread
from time import sleep

class draw:
  batch_list = []
  loss_list = []
  acc_list = []
  i = 0
  def __init__(self, pic_size):
    rcParams['figure.figsize']=pic_size, pic_size
    self.fig = plt.figure(figsize=(12, 4))
    self.batch_list = []
    self.train_loss_list = []
    self.train_acc_list = []
    self.i = 0
  def add(self, train_log):
    if len(self.batch_list) != 0:
      self.i = self.batch_list[-1] + 1
    else:
      self.i = self.i+1
    self.batch_list.append(self.i)
    self.train_loss_list.append(train_log[0])
    self.train_acc_list.append(train_log[1])
  def add_history(self, history):
      if len(self.batch_list) != 0:
          self.i = self.batch_list[-1] + len(history)
      else:
          self.i = self.i + len(history)
      self.batch_list.append(self.i)
      self.train_loss_list.append(history['loss'])
      self.train_acc_list.append(history['accuracy'])
  def save(self, NetPath):
    np.save(NetPath+'/b.npy', np.array(self.batch_list))
    np.save(NetPath+'/train_loss.npy', np.array(self.train_loss_list))
    np.save(NetPath+'/train_acc.npy', np.array(self.train_acc_list))
  def load(self, NetPath):
    self.batch_list = np.load(NetPath+'/b.npy').tolist()
    self.train_loss_list = np.load(NetPath+'/train_loss.npy').tolist()
    self.train_acc_list = np.load(NetPath+'/train_acc.npy').tolist()
  def close(self, time):
      sleep(time)
      plt.close()
  def show(self, file_path='E:/temp_pic/pic'):
    plt.clf()
    ax1 = self.fig.add_subplot(121)
    ax2 = self.fig.add_subplot(122)
    ax1.plot(self.batch_list, self.train_loss_list, label = 'train loss', color = 'red')
    ax2.plot(self.batch_list, self.train_acc_list, label = 'train acc', color = 'green')
    bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
    ax1.annotate("%s" % self.train_loss_list[-1], xy=(self.i, self.train_loss_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    plt.annotate("%s" % self.train_acc_list[-1], xy=(self.i, self.train_acc_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    ax1.set(xlabel='batches',ylabel='loss', title = 'loss')
    ax2.set(xlabel='batches',ylabel='acc', title = 'acc')

    plt.savefig(file_path+str(self.i) + '.png')
    # thread1 = Thread(target=self.close, args=(1,))
    # thread1.start()
    # plt.show()
  def show_images(self, images):
    for i in range(images.shape[0]):
      plt.subplot(1, images.shape[0], i + 1)
      plt.imshow(images[i])
      plt.axis('off')
    plt.tight_layout()
    plt.show()
  def show_image(self, image):
    plt.imshow(image)
    plt.show()