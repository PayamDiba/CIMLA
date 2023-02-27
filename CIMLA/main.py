import argparse
from CIMLA.ConfigParser import parser
"""
TODOs:
1. Move post processor to CIMLA_pipeline --> helps making it available as a module
"""

def main():

    flag = argparse.ArgumentParser()
    flag.add_argument('--config', type = str, default = None, help = 'path to config file', required = True)
    config_file = flag.parse_args().config
    CIMLA_pipeline, postProcessor, confs = parser.parse(config_file)
    CIMLA_pipeline.run(attr_data_split = confs['attribution']['data_split'],
                       attr_data_group = confs['attribution']['data_group'],
                       attr_data_size = confs['attribution']['data_size'],
                       global_type = confs['attribution']['global_type'])
    print("* Post-processing ...")
    postProcessor.process()
    #postProcessor.clear()

if __name__ == "__main__":
    main()
