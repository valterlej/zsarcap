
f = "/home/valter/Documentos/prop_results_val_1_e22_maxprop100_rgbonly.csv"

data = open(f,"r").readlines()

new_f = "/home/valter/Documentos/prop_results_val_1_e22_maxprop100_mdvc_rgbonly.csv"
new_f = open(new_f,"w")

for i, l in enumerate(data):
    l = l [:-1]
    l = l.split("\t")
    
    new_line = ""
    new_line += l[0] #video_id
    new_line += "\t"
    new_line += l[1] #caption_pred
    new_line += "\t"
    new_line += l[2] #start
    new_line += "\t"
    new_line += l[3] #end
    new_line += "\t"
    new_line += l[4] #duration
    new_line += "\t"
    if i == 0:
        new_line += "category_32"
    else:
        new_line += "1" #category_32
    new_line += "\t"
    new_line += "PLACEHOLDER"
    new_line += "\t"
    new_line += l[5] #phase
    new_line += "\t"
    new_line += l[6]
    new_line += "\n"
    new_f.write(new_line)

new_f.close()
