from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 暂时先这么用着，利用系统环境变量来存储ChatOpenAI实例化需要的key，base_URL等参数

load_dotenv()

models = {
    "DeepSeek_V3" : ChatOpenAI(model= "DeepSeek-V3", streaming = True),
    "DeepSeek_R1" : ChatOpenAI(model= "DeepSeek-R1", streaming = True)
}