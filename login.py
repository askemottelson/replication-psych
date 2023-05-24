import huggingface_hub as hh
import os

os.chdir("../")

file = open('hf.token', mode='r')
token = str(file.read()).strip()
file.close()

hh.login(token=token)
