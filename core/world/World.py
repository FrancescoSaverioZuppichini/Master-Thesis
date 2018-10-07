from pathlib import Path

class World():
    def __init__(self, name, format, base_dir='./worlds'):
        self.name = name
        self.base_dir = base_dir
        self.format = format

    @property
    def path(self):
        return Path("{}/{}.{}".format(self.base_dir,
                                          self.name,
                                          self.format))
