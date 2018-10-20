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

    def __call__(self, *args, **kwargs):
        """
        This method should initialise the world. E.g. open file,
        load information or load the map into the simulator
        :param args:
        :param kwargs:
        :return:
        """
        pass

    # TODO probably it is better to add a class method to CREATE the world from a file
    @property
    def path(self):
        return Path(self.file_path)

    @property
    def random_position(self):
        """
        This function must return a random position inside the world
        :return:
        """
        return None