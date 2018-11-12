from pathlib import Path


class World():
    """
    Basic abstraction of a world. It contains the basic information
    about its location in the file system.
    Ideally it should be able to load some information about it,
    for example the coordinates.
    """

    def __call__(self, *args, **kwargs):
        """
        This method should initialise the world. E.g. open file,
        load information or load the map.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @classmethod
    def from_file(cls, file_path):
        pass

    @property
    def random_position(self):
        """
        This function must return a random position inside the world
        :return:
        """
        return None