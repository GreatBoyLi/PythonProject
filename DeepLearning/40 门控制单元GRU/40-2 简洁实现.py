from torch import nn
from d2l import torch as d2l
from liwp import litorch as li

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size

gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
li.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

print(li.predict_ch8('time traveller', 50, model, vocab, device))
print(li.predict_ch8('traveller', 50, model, vocab, device))
print(li.predict_ch8('time', 50, model, vocab, device))
