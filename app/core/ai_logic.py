import os
from openai import OpenAI
from dotenv import load_dotenv

# 自动加载根目录下的 .env 文件
load_dotenv()

class YayoiBrain:
    def __init__(self):
        # 1. 这里的配置全部来自你刚刚写的 .env
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # 2. 初始化 OpenAI 兼容客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_visual_strategy(self, product_name, product_desc):
        """
        核心逻辑：根据产品信息，让 DeepSeek 返回视觉设计建议
        """
        system_prompt = (
            "你是一位资深的文创品牌视觉总监（弥生文创专用）。"
            "你的任务是根据产品特征，提供具体的背景生成提示词（Prompt）和布局建议。"
        )
        
        user_content = f"产品名称：{product_name}\n产品描述：{product_desc}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"DeepSeek 接口连接失败: {str(e)}"

# 测试代码（仅在直接运行此文件时执行）
if __name__ == "__main__":
    brain = YayoiBrain()
    print(brain.get_visual_strategy("樱花和纸胶带", "带有人间四月天意象的粉色半透明胶带"))
