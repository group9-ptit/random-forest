from core import helper


df = helper.read_csv("datasets/phishing.csv", unique_rows=True)
helper.write_csv(df, "datasets/phishing.csv")
