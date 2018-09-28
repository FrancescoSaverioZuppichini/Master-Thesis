class AgentCallback():
    def on_state_change(self, key, value):
        pass
    def on_subscribe(self, topic, data):
        pass

    def on_publish(self, topic, data):
        pass

    def on_shut_down(self):
        pass