"""
Run with `python -m Tools.ConfigOpener <filename>`
"""
import pickle as pkl
import argparse


def print_configuration(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    print(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    print_configuration(args.filename)


if __name__ == '__main__':
    main()
