import glob

source_file = "/Users/tannp3.aic/Transformer-MGK/dataset/*.*"
target_file = "/Users/tannp3.aic/Transformer-MGK/test.txt"
total_data = []
for i in glob.glob(source_file):
    with open(i, "r") as f:
        total_data.append(f.read().replace("\n\n","\n"))
    f.close()

with open(target_file,'w') as f:
    f.write("\n".join(total_data))