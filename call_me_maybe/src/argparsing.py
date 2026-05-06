from argparse import ArgumentParser

def arg_parsing():
    parser = ArgumentParser()

    parser.add_argument("--functions_definition", "-f", default="data/input/functions_definition.json")
    parser.add_argument("--input", "-i", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", "-o", default= "data/output/function_calling_tests.json")
    args = parser.parse_args()

    return [args.functions_definition, args.input, args.output]


