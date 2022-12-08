# Starter script
import time
import os

CC = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

def start():
  # import files form src folder and run them
  time.sleep(1)
  CC()
  from Src import blackJackNN


# Call to
start()