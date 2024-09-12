import pandas as pd

pd.set_option('display.max_rows', None)  # `None` means displaying all rows

df=pd.read_csv("/home/leon-gold/jepa/csvs/0/0simstep.csv")
df1=df["file_path"]
with open("csvs/0/0simstep.txt", "w") as file:
    file.write(str(df1))


print(df1)