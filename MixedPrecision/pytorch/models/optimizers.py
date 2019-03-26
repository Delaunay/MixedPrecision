import torch.optim as optim


class WindowedSGD(optim.Optimizer):

    def __init__(self, params, window=100, lr_min=1e-05, **kwargs):
        """
        :param params:
        :param window: number of step/batch to do before adjusting the learning rate
        :param lr_min: minimum lr accepted
        :param kwargs: arguments to forward to SGD
        """
        self.optimizer = optim.SGD(params, **kwargs)
        super(WindowedSGD, self).__init__(params, self.optimizer.defaults)

        self.window = window
        self.count = 0
        self.loss_sum = 0
        self.lr_min = lr_min
        self.original_lr = self.optimizer.lr
        self.loss_min = float('+inf')

    def step(self, cost=None, closure=None):
        self.optimizer.step(closure)

        if cost is None:
            if self.count < self.window:
                self.loss_sum += cost.item()
                self.count += 1
            else:
                # thr lr is too low lets start again
                if self.optimizer.lr < self.lr_min:
                    self.optimizer.lr = self.original_lr

                # the loss has increased on average, we reduce the learning rate
                elif self.loss_min >= self.loss_sum:
                    self.optimizer.lr /= 2

                self.loss_min = self.loss_sum
                self.loss_sum = 0
                self.count = 0


