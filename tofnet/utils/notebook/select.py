from collections import OrderedDict
from IPython.display import Markdown, display, JSON
import ipywidgets as widgets

from tofnet.utils.config import BaseConfig as Config

class SelectLogs:
    def __init__(self, logspath):
        self.logs = sorted([(x.name, x) for x in logspath.iterdir()], reverse=True)
        self.select = widgets.SelectMultiple(
            options=self.logs,
            rows=10,
            layout={'width':'100%'}
        )
        display(self.select)

    @property
    def value(self):
        return self.select.value

    def configs(self):
        config_list = []
        for _, logdir in self.logs:
            config_list.append(Config(configfile=logdir/"base_config.toml"))

        return config_list