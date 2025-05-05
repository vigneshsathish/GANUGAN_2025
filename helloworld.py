import pandas as pd
import numpy as np

df = pd.DataFrame({"Id":[1,2,3],
                  "Name":["Viky","Sathish",np.nan]
                })
print(df)