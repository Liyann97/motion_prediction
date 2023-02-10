import configparser
import os


def get_configuration():
    proDir = os.path.split(os.path.realpath(__file__))[0]
    # print(proDir)
    configPath = os.path.join(proDir, "config.txt")
    # path = os.path.abspath(configPath)
    # print(configPath)
    # print(path)

    conf = configparser.ConfigParser()
    conf.read(configPath)

    window = conf.getint('hyperParameter', 'window')
    lag = conf.getint('hyperParameter', 'lag')
    input_index = conf.getint('hyperParameter', 'input_index')
    output_index = conf.getint('hyperParameter', 'output_index')

    # print(type(window))
    # print(lag)
    # print(type(input_index))
    # print(output_index)
    return window, lag, input_index, output_index
