import sys,os

last_key = ""
last_handle = None
weight = ""
for line in sys.stdin:
    key, handle = line.strip().split('\t')[:2]
    cur_weight = "\t".join(line.strip().split('\t')[2:])
    if last_handle is None:
        last_key = key
        last_handle = int(handle)
        weight = cur_weight
        continue
    if last_key == key:
        if int(handle) - last_handle != 1:
            sys.stderr.write("input format error, key[{}], last_handle[{}], handle[{}]\n".format(key, last_handle, handle))
            last_key = ""
            last_handle = None
            continue
        weight = weight + "\t" + cur_weight
        last_handle = int(handle)
    else:
        sys.stdout.write("{}\t{}\n".format(last_key, weight))
        last_key = key
        last_handle = int(handle)
        weight = cur_weight

        
sys.stdout.write("{}\t{}\n".format(last_key, weight))
