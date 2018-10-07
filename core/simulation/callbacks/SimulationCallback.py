class SimulationCallback():
    def on_start(self, sim, world, agent, *args, **kwargs):
        pass

    def on_finish(self, sim, world, agent, *args, **kwargs):
        pass

    def tick(self, sim, world, agent, *args, **kwargs):
        pass