import huggingface_hub as hh

file = open('/home/asmo/hf.token', mode='r')
token = str(file.read()).strip()
file.close()

hh.login(token=token)