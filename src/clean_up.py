from pathlib import Path
import pandas as pd

# clean up artificats of the translation API

path = Path("data/dev/europarl_dev.csv")

data = pd.read_csv(path, index_col=0, header = 0) 

data = data.dropna()
data['Original'] = data['Original'].apply(lambda x: str(x).strip('\n'))
data['Human'] = data['Human'].apply(lambda x: str(x).strip('\n'))
data['Automated'] = data['Automated'].apply(lambda x: str(x).replace('&quot;', '\"').replace('&#39;', '\'').strip('\n'))

data.to_csv(path)