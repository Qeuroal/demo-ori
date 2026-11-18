# coding=utf-8

__all__ = ["PRINT_INFO", "PRINT_WARNING", "PRINT_ERROR",
           "PRINT_INFOES", "PRINT_WARNINGS", "PRINT_ERRORS"]

def PRINT_ERROR(*args, fin=False, **kwargs):
    filename, line, end, sep = __parse_info(kwargs)
    print("\033[31m==>>> {0}{2}{1}{3} "
          .format(filename, line, ":" if (filename and line) else "", ":" if (filename or line) else ""), *args, "\033[0m", sep=sep, end=end)
    if (fin): exit(0)


def PRINT_ERRORS(*args, fin=False, **kwargs):
    filename, line, end, sep = __parse_info(kwargs)
    sep = kwargs.get("sep", "\033[0m\n\033[34m" + ("-" * 64) + "\n\033[0m\033[31m")
    print("\033[31mvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv{0}{2}{1}vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
          .format(filename, line, ":" if (filename and line) else ""))
    print(*args, sep=sep)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^{0}{2}{1}^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\033[0m"
          .format(filename, line, ":" if (filename and line) else ""), end=end)
    if (fin): exit(0)


def PRINT_WARNING(*args, fin=False, **kwargs):
    filename, line, end, sep = __parse_info(kwargs)
    print("\033[33m==>>> {0}{2}{1}{3} "
          .format(filename, line, ":" if (filename and line) else "", ":" if (filename or line) else ""), *args, "\033[0m", sep=sep, end=end)
    if (fin): exit(0)


def PRINT_WARNINGS(*args, fin=False, **kwargs):
    filename, line, end, sep = __parse_info(kwargs)
    sep = kwargs.get("sep", "\033[0m\n\033[34m" + ("-" * 64) + "\n\033[0m\033[33m")
    print("\033[33mvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv{0}{2}{1}vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
          .format(filename, line, ":" if (filename and line) else ""))
    print(*args, sep=sep)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^{0}{2}{1}^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\033[0m"
          .format(filename, line, ":" if (filename and line) else ""), end=end)
    if (fin): exit(0)


def PRINT_INFO(*args, fin=False, **kwargs):
    filename, line, end, sep = __parse_info(kwargs)
    print("\033[32m==>>> {0}{2}{1}{3} "
          .format(filename, line, ":" if (filename and line) else "", ":" if (filename or line) else ""), *args, "\033[0m", sep=sep, end=end)
    if (fin): exit(0)


def PRINT_INFOES(*args, fin=False, **kwargs):
    filename, line, end, sep = __parse_info(kwargs)
    sep = kwargs.get("sep", "\033[0m\n\033[34m" + ("-" * 64) + "\n\033[0m\033[32m")
    print("\033[32mvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv{0}{2}{1}vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
          .format(filename, line, ":" if (filename and line) else ""))
    print(*args, sep=sep)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^{0}{2}{1}^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\033[0m"
          .format(filename, line, ":" if (filename and line) else ""), end=end)
    if (fin): exit(0)


def __parse_info(kwargs):
    filename = kwargs.get("filename", "")
    line = kwargs.get("line", "")
    end = kwargs.get("end", "\n")
    sep = kwargs.get("sep", " ")

    return filename, line, end, sep


if __name__ == "__main__":
    PRINT_ERROR("hello")

