import pandas as pd
from sklearn.model_selection import StratifiedKFold

random_state = 21

path = 'data/train_spleen.csv'
df = pd.read_csv(path)
df['fold'] = 0

skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(df, df.high)):
    print(f"Fold {i}:")
    ids = df.loc[test_index].patient_id.tolist()
    df['fold'] = df.apply(lambda x: i if x.patient_id in ids else x.fold, axis=1)

df.to_csv(path, index=False)