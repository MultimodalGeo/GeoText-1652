import csv
import ast
import json
import os
# Open the original csv file
data = []
for i in range(0,900):
    s = "%04d" % i
    path = s+'/'+'apt.csv'
    if os.path.exists(path)==False:
        print(s)
        continue
    with open(path, "r") as file:
        s = "%04d" % i
        reader = csv.reader(file)
        for row in reader:
            # Replace all single quotes with double quotes in each row
            if row == ['image', 'prompt']:
                continue
            else:
                # if row[0][-4:] == "jpeg":


                    # print(s)
                new_row_0 = [cell.replace('"', "\'") for cell in row]
                new_row = [cell.replace("['", "[\"") for cell in new_row_0]
                new_row_1 = [cell.replace("']", '\"]') for cell in new_row]
                new_row_2 = [cell.replace("', '", "\", \"") for cell in new_row_1]
                new_row_3 = [cell.replace("\", '", "\", \"") for cell in new_row_2]
                new_row_4 = [cell.replace("', \"", "\", \"") for cell in new_row_3]

                    #Test&Valid
                    # p = new_row_4[1]
                    # k = ast.literal_eval(p)
                    # print(k)
                new_dict = {}
                new_dict['image_id'] = s+'/'+new_row_4[0]
                new_dict['image'] = 'test'+'/'+'query'+'/'+ s+'/'+new_row_4[0]
                new_dict['caption'] = ast.literal_eval(new_row_4[1])
                data.append(new_dict)

                    #Train

# Write the modified data to a new csv file
with open("test_set.json", "w") as file:
    json.dump(data, file)