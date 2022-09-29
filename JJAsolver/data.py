
from datetime import datetime
import pathlib
import shutil
import os.path
import sys
import json

class datafolder:
    def __init__(
            self,
            folder,
            args={}
    ):
        # make new datafolder
        date_string = datetime.now()
        data_folder = date_string.strftime("%Y-%m-%d_%H-%M-%S")
        
        data_folder = data_folder + "_" + folder

        self.folder = data_folder
        print("creating new datafolder: ", data_folder)
        pathlib.Path(data_folder).mkdir()
        script = sys.argv[0]
        print("script: ", script)
        
        # copy script into
        # copy script into output folder
        shutil.copyfile(script, data_folder + '/' + os.path.basename(script))
                
        # generate metadata file
        # copy script args into metadata

        with open(data_folder + '/args.json', 'w') as outfile:
            json.dump(args, outfile)
        

class datafile:
    def __init__(
            self, folder,
            file : str = "data.dat",
            params = [],
            number_format = '%.10g',
            number_delimiter = "\t\t",
           
    ):
        self.number_format = number_format
        self.number_delimiter = number_delimiter
        

        

        # generate datafile
        data_filename = folder.folder + "/" + file
        self.datafile = open(data_filename, "w")
        self.params = params

        # write datafile header
        header = "# " + number_delimiter.join(params) + "\n"
        self.datafile.write(header)

        self.datafile.flush()
        os.fsync(self.datafile)

        
    def log(self, params={}):
        number_format = self.number_format
        number_delimiter = self.number_delimiter
        
        params_list = []
        values = []
        line = ''
        for param in self.params:
            if not param in params:
                print("missing key")
                raise ValueError("missing key in params: %s" % param)
            values.append(number_format % params[param])
        line = number_delimiter.join(values) + "\n"

        self.datafile.write(line)

        # flush buffers
        self.datafile.flush()
        os.fsync(self.datafile)
            
    def new_block(self):
        self.datafile.write("\n")
        # flush buffers
        self.datafile.flush()
        os.fsync(self.datafile)
        
            
                
        
        
        
        
        
        
        
