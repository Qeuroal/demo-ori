# coding=utf-8
from prettytable import PrettyTable

class PrettyPrint(object):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        pt = PrettyTable()
        header = ["", "value"]
        pt.field_names = header

        for var in vars(self):
            pt.add_row([repr(var), "{}".format(eval("self.{}".format(var)))])

        return str(pt)
