import csv
import ast
import json
import os
import re
# Open the original csv file

def remove_sentences_containing_word(text, word):
    sentences = text.split('.')
    # print(sentences)
    sentences_without_word = []
    for sen in sentences:
        if word not in sen:
            sentences_without_word.append(sen)

    # print(sentences_without_word)
    new_text = '.'.join(sentences_without_word)
    return new_text

def remove_sentence_less(text, n):
    sentences = text.split('.')
    # print(sentences)
    sentences_without_word = []
    for sen in sentences:
        if len(sen.split(" "))>n:
            sentences_without_word.append(sen)

    # print(sentences_without_word)
    new_text = '.'.join(sentences_without_word)
    return new_text



def insert_period(text):
    # 正则表达式找出小写字母后面紧跟的大写字母
    pattern = r'([a-z])([A-Z])'

    # 在小写字母和大写字母之间插入句点和空格
    modified_text = re.sub(pattern, r'\1. \2', text)
    
    return modified_text

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
                    # 找到最后一个句号的位置
                    last_period_idx = caption.rfind('.')

                    # 如果找到句号，保留该位置之前的所有文本（不包括句号之后的内容）
                    if last_period_idx != -1:
                        new_caption = caption[:last_period_idx+1]
                    
                    # findLast = new_caption.rfind('center')
                    # if findLast != -1:
                    #     N_caption = new_caption[:findLast+1]
                    # fine = N_caption.rfind('.')
                    # if fine != -1:
                    #     fine_caption = new_caption[fine+1:]

                    # 使用 split 方法将文本划分为句子
                    sentences = new_caption.split('.')
                    sea = []

                    for sen in sentences:
                        sea.append(insert_period(sen))
                    
                    new_caption = ".".join(sea)

                    new_cap = remove_sentences_containing_word(new_caption, "https")

                    new_cap = remove_sentences_containing_word(new_cap, "overall")

                    new_cap = remove_sentences_containing_word(new_cap, "Overall")                   

                    new_cap = remove_sentence_less(new_cap, 9)



                    sentences = new_cap.split('.')

                    # 只保留前三个句子，并使用 '.' 连接这三个句子
                    new_caption = '.'.join(sentences[-5:])

                    new_caption = new_caption+"."

                    prompt_new.append(new_caption)

                prompts.append(prompt_new)

        print('Finish!')

        if len(prompts):
            csv_path = os.path.join(folder_path, 'apt_last.csv')
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(['image', 'prompt'])
                # print(files)
                # print(prompts)
                for file, prompt in zip(files, prompts):
                    w.writerow([file, prompt])
                    # print([file, prompt])


            print(f"\n\n\n\nGenerated {len(prompts)} prompts and saved to {csv_path}, enjoy!")
        else:
            print(f"Sorry, I couldn't find any images in {folder_path_now}")

