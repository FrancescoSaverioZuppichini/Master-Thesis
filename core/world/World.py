from pathlib import Path


class World():
    """
    Basic abstraction of a world. It contains the basic information
    about its location in the file system.
    Ideally it should be able to load some information about it,
    for example the coordinates.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    @property
    def path(self):
        return Path(self.file_path)

    def __call__(self, *args, **kwargs):
        """
        TODO load information about the world
        :param args:
        :param kwargs:
        :return:
        """
        pass
