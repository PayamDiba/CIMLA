"""
TODOs:
1. organize key default params
"""
default_params = {'batch':-1,
                  'shuffle':False,}

"""
functions
"""

def params(config):
    for k in default_params.keys():
        if k not in config.keys():
            config[k] = default_params[k]

    return config
