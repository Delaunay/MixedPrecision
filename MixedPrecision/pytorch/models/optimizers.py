import torch.optim as optim


class WindowedSGD:

    def __init__(self, params, window=100, lr_min=1e-05, lr=0.01, epoch_steps=None, **kwargs):
        """
        :param params:
        :param window: number of step/batch to do before adjusting the learning rate
        :param lr_min: minimum lr accepted
        :param kwargs: arguments to forward to SGD
        """
        self.optimizer = optim.SGD(params, lr, **kwargs)

        self.window = window
        self.count = 0
        self.loss_sum = 0
        self.lr_min = lr_min
        self.original_lr = lr
        self.reset_count = 1            # Use for annealing the jumps are smaller and smaller
        self.loss_min = float('+inf')

        self.epoch_steps = epoch_steps
        self.all_cost = 0
        self.all_count = 0
        self.lr = lr

    def set_lr(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, cost=None, closure=None):
        self.set_lr()
        self.optimizer.step(closure)

        # compute the epoch cost
        if self.epoch_steps is not None:
            if self.all_count > self.epoch_steps:
                self.all_count = 0
                self.all_cost = 0

            self.all_cost += cost.item()
            self.all_count += 1

        # Compute the Window cost
        if cost is not None:
            if self.count < self.window:
                self.loss_sum += cost.item()
                self.count += 1
            else:
                # thr lr is too low lets start again
                if self.lr < self.lr_min:
                    self.lr = self.original_lr / self.reset_count
                    self.reset_count += 1

                # the loss has increased on average, we reduce the learning rate
                elif self.loss_sum > self.loss_min:
                    self.lr /= 2

                self.loss_min = self.loss_sum
                self.loss_sum = 0
                self.count = 0


