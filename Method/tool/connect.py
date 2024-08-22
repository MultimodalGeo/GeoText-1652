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
    folder_path_now = folder_path+'/'+'apt_three.csv'
    folder_path___ = folder_path+'/'+'apt_last.csv'

    files=[]
    prompts=[]
    if os.path.exists(folder_path_now)==False:
        print(folder_path_now)
        print(s)
        continue
    with open(folder_path_now, "r") as file:
        s = "%04d" % i
        reader = csv.reader(file)
        with open(folder_path___, "r") as file1:
            reader1 = csv.reader(file1)
            file1_rows = list(reader1)

        for row in reader:
            # Replace all single quotes with double quotes in each row
            if row == ['image', 'prompt']:
                continue
            else:
                file = row[0]
                print(file)
                for row_ in file1_rows:
                    # Replace all single quotes with double quotes in each row
                    if row_[0] == file:
                        print(row_[0])
                        ppp = row_[1]
                        print(ppp)
                files.append(file)
                prompt = ast.literal_eval(row[1])
                prompt_ = ast.literal_eval(ppp)
                prompt_new=[]
                for i in range(3):
                    kkk = prompt[i] + prompt_[i]

                    prompt_new.append(kkk)
                    
                prompts.append(prompt_new)

        print('Finish!')

        if len(prompts):
            csv_path = os.path.join(folder_path, 'apt_new.csv')
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
