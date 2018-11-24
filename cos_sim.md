Since the code is relatively easy to write, I've shared it below. 

```python
from scipy import spatial
import numpy as np 

similarities = []
for text in myarray:
    score = 1 - spatial.distance.cosine(list(text), list(myarray[125]))
    if np.isnan(score):
        score = 0.0
    similarities.append(score)

df_sim = pd.DataFrame(similarities, columns=["cos_sim"])
df_sim['filenames'] = output_filenames 
df_sim = df_sim.sort_values(by="cos_sim", ascending=False).reset_index(drop=True)
df_sim
```

Note that the above code is dependent on ...