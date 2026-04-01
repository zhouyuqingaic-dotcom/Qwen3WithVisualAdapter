from openai import OpenAI
import time

class GPT5MiniClient:
    """
    GPT-5-Mini (AGICTO API) 的极简封装类。
    职责：纯粹的 API 请求与重试机制，不包含任何业务和解析逻辑。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-5-mini"):
        if not api_key:
            raise ValueError("API Key 不能为空，请检查配置文件！")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def ask(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, retries: int = 3) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                )
                return chat_completion.choices[0].message.content.strip()

            except Exception as e:
                print(f"⚠️ [API 请求报错] 第 {attempt + 1} 次尝试失败: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    print("❌ 达到最大重试次数，放弃请求。")
                    return "ERROR"