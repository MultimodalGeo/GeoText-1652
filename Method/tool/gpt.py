import requests

def call_gpt_api(prompt, api_key):
    url = "https://api.openai.com/v1/engines/davinci-codex/completions"  
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 100  
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 使用 API
api_key = "<YOUR_API_KEY>"  
prompt = "Translate the following English text to French: 'Hello, how are you?'"
result = call_gpt_api(prompt, api_key)

print(result)


import requests

def check_text_requirements(prompt, api_key):
    url = "https://api.openai.com/v1/engines/davinci-codex/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 50  # 根据需要调整
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 使用 API
api_key = "<YOUR_API_KEY>"  # 替换为你的 API 密钥
text_to_check = "Your text here"
prompt = f"Please check if the following text is a description of some buildings: '{text_to_check}', if correct, say why it is correct, if wrong, say why it is wrong."
result = check_text_requirements(prompt, api_key)

print(result)
