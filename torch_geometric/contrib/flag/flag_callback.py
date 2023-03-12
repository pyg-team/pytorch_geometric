class FLAGCallback:
    def on_ascent_step_begin(self, step_index, loss):
        pass

    def on_ascent_step_end(self, step_index, loss, perturb_data):
        pass

    def on_optimizer_step_begin(self, loss):
        pass

    def on_optimizer_step_end(self, loss):
        pass


class FLAGLossHistoryCallback(FLAGCallback):
    def __init__(self):
        self.loss_history = []

    def on_ascent_step_end(self, step_index, loss, perturb_data):
        self.loss_history.append(loss)


class FLAGPerturbHistoryCallback(FLAGCallback):
    def __init__(self):
        self.perturb_history = []

    def on_ascent_step_end(self, step_index, loss, perturb_data):
        self.perturb_history.append(perturb_data.cpu().detach().numpy())
