def typeInit(df):
    cols = df.columns
    res = {}
    for col in cols:
        if str(df[col].dtype).startswith('int') or str(df[col].dtype).startswith('float'):
            uni = df[col].nunique()
            if uni < min(df.shape[0]//10, 10000):
                res[col] = 'cat'
                continue
        res[col] = 'num'
    return res