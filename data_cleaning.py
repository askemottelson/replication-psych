import re
from string import digits, punctuation

class DataObject:
  
  def __init__(self, train_df, train_texts):
    self.train_df = train_df
    self.train_texts = train_texts

  def clean(self, row):
    remove_digits = str.maketrans('', '', digits)
    remove_punc = str.maketrans('','', punctuation)

    text = row.full_text
    #text = text.translate(remove_punc)
    text = text.translate(remove_digits)
    stats = ["p","r","P","rho","ρ","Χ","ω","f","F","n","N","β","CI"]

    for stat in stats:
      expr = re.compile(r'\b' + re.escape(stat) + r'\b', re.IGNORECASE)
      text = expr.sub("", text)

    signs = ["(",")","<",">","=","±",".,","..", "  "]
    for sign in signs:
      text = text.replace(sign, "")
    
    return text

  # todo: count these per doi, and take majority vote as training label
  def compute_label(self, row):
    doi = row.doi
    replications = self.train_df[self.train_df['doi'] == doi]
    false_count = len(replications[replications['replicated_binary'] == 'no'])
    true_count = len(replications[replications['replicated_binary'] == 'yes'])

    if false_count + true_count == 0:
      return "error"
    if true_count == false_count:
      return "maybe"
    elif true_count > false_count:
      return "yes"
    else:
      return "no"

  def modify_data(self):
    print("hej")
    self.train_texts['label'] = self.train_texts.apply(self.compute_label, axis=1)
    self.train_texts['full_text'] = self.train_texts.apply(self.clean, axis=1)

    data = self.train_texts[self.train_texts['label'].isin(["yes", "no"])] # binary replication

    return data


##################################################################

# utilization function
def gpu_utilization():
  nvmlInit()
  handle = nvmlDeviceGetHandleByIndex(0)
  info = nvmlDeviceGetMemoryInfo(handle)
  print(f"Used GPU memory so far: {info.used//1024**2} MB")
