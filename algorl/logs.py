
import logging
from rich.logging import RichHandler

# set up logging to file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d-%m-%Y %H:%M',
                    filename='algorl_costume_algos.log',
                    filemode='w'
                    )

# Define a handler which writes INFO messages or higher
console = RichHandler()
console.setLevel(logging.INFO)

# Set a format which is simpler for console use
formatter = logging.Formatter('[%(asctime)s] %(name)-12s: %(levelname)-6s: %(message)s', datefmt='%H:%M',)

# Tell the handler to use this format
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

# Disable messages from external libraries lower the warnings
logging.getLogger("matplotlib").setLevel(logging.WARNING)