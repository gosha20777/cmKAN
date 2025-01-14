import argparse
from cm_kan.core import logger
from cm_kan import cli


def parse_arguments() -> argparse.Namespace:
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        'cmKAN CLI', 
        formatter_class=cli.RichHelpFormatter
    )
    subparser = parser.add_subparsers(title='Tools', required=True)

    cli.register_parsers(subparser)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.func(args)


if __name__ == '__main__':
    main()
