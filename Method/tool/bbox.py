import csv
import ast
import os
import json
# Open the original csv file
data = []
folder_path = "/storage_fast/mchu/Multi-model/geotext/GeoText/train"
n = 0
for i in range(800,1670):
    s = "%04d" % i
    path = folder_path + '/' + s +'/'+'apolla_new.csv'
    if os.path.exists(path)==False:
        print(s)
        continue
    with open(path, "r") as file:
        s = "%04d" % i
        reader = csv.reader(file)
        for row in reader:
            # Replace all single quotes with double quotes in each row
            if row == ['image', 'REC']:
                continue
            elif row[1] == []:
                continue
            else:
                if row[0][-4:] == "jpeg":
                    # if 'None' in row[1]:
                    #     continue
                    print(s)
                    new_row_0 = [cell.replace('"', "\'") for cell in row]
                    new_row = [cell.replace("['", "[\"") for cell in new_row_0]
                    new_row_1 = [cell.replace("']", '\"]') for cell in new_row]
                    new_row_2 = [cell.replace("', '", "\", \"") for cell in new_row_1]
                    new_row_3 = [cell.replace("\", '", "\", \"") for cell in new_row_2]
                    new_row_4 = [cell.replace("', \"", "\", \"") for cell in new_row_3]

                    sens = ast.literal_eval(new_row_4[1])

                    for new in sens:
                        # new_ = new.replace("Description:",'')
                        # string = new.replace("'object'", '"object"')
                        # string_ = string.replace("'bbox'",'"bbox"')
                        # string__= string_.replace("'phrases'",'"phrases"')

                        # string___ = string__.replace("\n\n","")
                        # print(string___)
                        # old_dict = json.loads(string___)

                        objects = new['description']
                        bboxes = new['bbox']

                        # if len(objects)<3 or len(bboxes)<3:
                        #     n = n+1
                        #     continue 

                        #Test&Valid
                        # p = new_row_4[1]
                        # k = ast.literal_eval(p)
                        # print(k)
                        # new_dict = {}
                        # new_dict['image_id'] = s+'/'+new_row_4[0]
                        # new_dict['image'] = 'Test'+'/'+ s+'/'+new_row_4[0]
                        # new_dict['caption'] = ast.literal_eval(new_row_4[1])
                        # data.append(new_dict)

                        #Train
                        # captions = ast.literal_eval(new_row_4[1])
                        # for caption in captions:
                        #     new_dict = {}
                        #     new_dict['image_id'] = s+'/'+new_row_4[0]
                        #     new_dict['image'] = 'Train'+'/'+ s+'/'+new_row_4[0]
                        #     new_dict['caption'] = caption
                        #     data.append(new_dict)


                        #OBJECT
                        obj = []
                        tb = ["main", "left", "right"]
                        for i in range(3):
                            A = {}
                            t = tb[i]
                            A[t] = objects[i]
                            A["bb"] = bboxes[i]
                            print(bboxes[i])
                            obj.append(A)

                        new_dict = {}
                        new_dict["obj"] = obj
                        new_dict["image_id"] = s+'/'+new_row_4[0]
                        new_dict["image"] = 'Train'+'/'+ s+'/'+new_row_4[0]
                        data.append(new_dict)


print("The number of n is", n)

# Write the modified data to a new csv file
with open("bbox-new.json", "w") as file:
    json.dump(data, file)