import yaml

class ConfigFile:

    def __init__(self,filename):
        self._load_config(filename)

    def __getitem__(self,key):
        try:
            return self.config[key]
        except KeyError:
            return None
        
    def _load_config(self,filename):
        try:
            file = open(filename, 'r')
        except OSError as e:
            print('open of ',filename,' failed', e)
            raise e

        try:
            self.config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    def print(self):
        for key in self.config:
            print(key,': ',self.config[key])
    

