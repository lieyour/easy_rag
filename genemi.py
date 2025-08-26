import google.generativeai as genai
import os

# 设置代理
proxy_url = "http://127.0.0.1:7897"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# 配置Gemini API
genai.configure(
    api_key="AIzaSyAKczj1Zqb7r_MNmUux_V2LrXj3RFGqtXo",
    transport='rest'
)

# 初始化模型
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.7,
        "max_output_tokens": 1000,
    }
)

# 开始对话
chat = model.start_chat(history=[])
response = chat.send_message("你好，我叫Wayne")
print(response.text)
