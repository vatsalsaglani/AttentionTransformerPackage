class ScheduledOptimizer():

    def __init__(self, optimizer, init_lr, dim_model, n_warmup_steps = 4000):

        self._optimizer = optimizer
        self.init_lr = init_lr
        self.dim_model = dim_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.state_dict = self.get_optimizer_state_dict()
        

    def step_and_update_lr(self):

        ''' Step with innner optimizer '''

        self._update_learning_rate()
        self._optimizer.step()

    def get_optimizer_state_dict(self):
        return self._optimizer.state_dict()


    def zero_grad(self):

        self._optimizer.zero_grad()

    def _get_lr_scale(self):

        d_model = self.dim_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps

        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):

        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr