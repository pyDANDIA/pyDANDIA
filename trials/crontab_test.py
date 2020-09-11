from sys import argv
from datetime import datetime

output_file = argv[1]

f = open(output_file, 'w')
f.write( 'Testing pyDANDIA under crontab '+datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')+'\n' )

from pyDANDIA import pipeline_setup

f.write('Completed import of pyDANDIA module')
f.close()
