import csv
import ast
import json
import os
# Open the original csv file
data = []
for i in range(0,1700):
    s = "%04d" % i
    # path = s+'/'+'apt.csv'
    folder_path = "/storage_fast/mchu/Multi-model/geotext/GeoText/test/gallery_no_train" +"/"+ s
    folder_path_now = folder_path+'/'+'apt.csv'

    files=[]
    prompts=[]
    if os.path.exists(folder_path_now)==False:
        print(folder_path_now)
        print(s)
        continue
    with open(folder_path_now, "r") as file:
        s = "%04d" % i
        reader = csv.reader(file)
        for row in reader:
            # Replace all single quotes with double quotes in each row
            if row == ['image', 'prompt']:
                continue
            else:
                # if row[0][-4:] == "jpeg":
                    # print(s)


                p = row[1].replace("[MOF]","")
                t = p.replace("\\n","")
                k = t.replace('[','')
                l = k.replace(']','')
                l = l.replace("\\",'')
                e = '['+l+']'
                row[1] = e

                new_row_0 = [cell.replace('"', "\'") for cell in row]
                new_row = [cell.replace("['", "[\"") for cell in new_row_0]
                new_row_1 = [cell.replace("']", '\"]') for cell in new_row]
                new_row_2 = [cell.replace("', '", "\", \"") for cell in new_row_1]
                new_row_3 = [cell.replace("\", '", "\", \"") for cell in new_row_2]
                new_row_4 = [cell.replace("', \"", "\", \"") for cell in new_row_3]

                #Train
                file = row[0]
                # print(new_row_4[1])
                # p = row[1].replace("[MOF]","")
                # t = p.replace("\\n","")
                # k = t.replace('[','')
                # l = k.replace(']','')
                # l = l.replace("\\",'')
                # e = '['+l+']'



                # print(e)
                files.append(file)
   
                e = new_row_4[1]
                prompt = ast.literal_eval(e)
                prompt_new=[]

                for caption in prompt:
                    # 使用 split 方法将文本划分为句子
                    sentences = caption.split('.')

                    # 只保留前三个句子，并使用 '.' 连接这三个句子
                    new_caption = '.'.join(sentences[:3]) + '.'

                    prompt_new.append(new_caption)
                prompts.append(prompt_new)

        print('Finish!')

        if len(prompts):
            csv_path = os.path.join(folder_path, 'apt_three.csv')
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(['image', 'prompt'])
                print(files)
                print(prompts)
                for file, prompt in zip(files, prompts):
                    w.writerow([file, prompt])


            print(f"\n\n\n\nGenerated {len(prompts)} prompts and saved to {csv_path}, enjoy!")
        else:
            print(f"Sorry, I couldn't find any images in {folder_path_now}")
  