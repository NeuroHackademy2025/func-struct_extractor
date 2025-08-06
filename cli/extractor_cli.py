import argparse
from func_struct_extractor.app import cli as fcli
if __name__ == "__main__":
    fcli(['extractor'] + sys.argv[1:])