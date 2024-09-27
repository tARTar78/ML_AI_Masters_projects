
        
def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    
    return df

def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        
        chunks.append(df)
        
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    
    return df
    
def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
    )
        
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
        
    #df_base = df_base.pipe(Pipeline.handle_dates)
    
    return df_base
def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data, cat_cols

def add_tax(train_data, s1,s2,s3):
  res = None
  i = 0
  for s0 in [s1,s2,s3]:
    i+=1
    directory = "/kaggle/input/aim-2024-local-contest-home-credit1/parquet_files/"
    research = pl.read_parquet(s0).pipe(Pipeline.set_table_dtypes)
    columns = research.columns
    a = research.select("case_id").unique()
    s = "taxes"
    a = a.join(research.group_by("case_id").agg(pl.col("case_id").count().alias(s+'_cnT')),how="left", on="case_id")
    for col in columns:
      if col.endswith('A') or col.endswith('L') or col.endswith('P') :
          a = a.join(research.group_by("case_id").agg(pl.col(col).max().alias("a"+s  +'_maX')),how="left", on="case_id")
          a = a.join(research.group_by("case_id").agg(pl.col(col).mean().alias("a"+s +'_meaN')),how="left", on="case_id")
    for col in columns:
      if col.endswith('T') or col.endswith('D'):
          a = a.join(research.group_by("case_id").agg(pl.col(col).max().alias("d"+s + '_firs' + col[-1])),how="left", on="case_id")
    for col in columns:
      if col.endswith('1') or col.endswith('2'):
          a = a.join(research.group_by("case_id").agg(pl.col(col).max().alias("g"+s +'_maX')),how="left", on="case_id")
          a = a.join(research.group_by("case_id").agg(pl.col(col).count().alias("g"+s +'_counT')),how="left", on="case_id")
    if (i == 1):
        res = a
    else:
        res = pl.concat([res,a], how="vertical_relaxed")
    del a, research
  train_data = train_data.join(
      res, how="left", on="case_id"
  )
  return train_data
