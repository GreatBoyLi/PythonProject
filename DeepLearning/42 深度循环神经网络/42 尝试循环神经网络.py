import torch
from torch import nn
from d2l import torch as d2l
from liwp import litorch as li

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
# num_layers 设置隐藏层
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, vocab_size)
model.to(device)

num_epochs, lr = 500, 2
li.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
print("**************************************************")
print(li.predict_ch8('time traveller', 50, model, vocab, device))
print(li.predict_ch8('traveller', 50, model, vocab, device))
print(li.predict_ch8('time', 50, model, vocab, device))
