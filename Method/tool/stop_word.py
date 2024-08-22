# 示例的英文停用词列表
stop_words = set(["a", "an", "the", "is", "at", "which", "on", "for", "with", "without", "and", "or", "in", "to", "from"])

def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# 测试文本
text = "The quick brown fox jumps over the lazy dog"
filtered_text = remove_stop_words(text)
print(filtered_text)
