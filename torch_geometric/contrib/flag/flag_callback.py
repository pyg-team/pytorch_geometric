class FLAGCallback:
  def on_ascent_step_begin(self, step_index, loss):
    pass

  def on_ascent_step_end(self, step_index, loss, perturb_data):
    pass

  def on_optimizer_step_begin(self, loss):
    pass

  def on_optimizer_step_end(self, loss):
    pass


