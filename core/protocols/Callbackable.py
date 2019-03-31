class Callbackable():
    def set_callbacks(self, callbacks):
        self.callbacks = callbacks
        return self

    def add_callback(self, callback):
        self.callbacks.append(callback)
        return self

    def add_callbacks(self, callbacks):
        for cb in callbacks:
            self.add_callback(cb)
        return self

    def notify(self, event_name, *args, **kwargs):
        for hook in self.callbacks:
            getattr(hook, event_name)(*args, **kwargs)
        return self