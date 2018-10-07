class Callbackable():
    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def add_callbacks(self, callbacks : list):
        for cb in callbacks:
            self.add_callback(cb)

    def notify(self, event_name, *args, **kwargs):
        for hook in self.callbacks:
            getattr(hook, event_name)(*args, **kwargs)
