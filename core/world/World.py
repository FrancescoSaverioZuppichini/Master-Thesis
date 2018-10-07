from pathlib import Path


class World():
    """
    Basic abstraction of a world. It contains the basic information
    about its location in the file system.
    Ideally it should be able to load some information about it,
    for example the coordinates.
    """

    def __init__(self, name, format, base_dir='./worlds'):
        self.name = name
        self.base_dir = base_dir
        self.format = format

    @property
    def path(self):
        return Path("{}/{}.{}".format(self.base_dir,
                                      self.name,
                                      self.format))

    def __call__(self, *args, **kwargs):
        """
        TODO load information about the world
        :param args:
        :param kwargs:
        :return:
        """
        pass
