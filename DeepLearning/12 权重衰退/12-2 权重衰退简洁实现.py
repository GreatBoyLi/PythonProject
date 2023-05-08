import torch
from torch import nn
from d2l import torch as d2l
import liwp.litorch as liwp


n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, ture_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, ture_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, ture_b, n_test)
test_iter = d2l.load_array(test_data,batch_size, is_train=False)


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for parm in net.parameters():
        parm.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([{
        "params": net[0].weight,
        "weight_decay": wd}, {
        "params": net[0].bias}], lr=lr)

    animator = liwp.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                             ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if(epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (liwp.evaluate_loss(net, train_iter, loss),
                                     liwp.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', net[0].weight.norm().item())


train_concise(10)
