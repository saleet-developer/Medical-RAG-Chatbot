from ctransformers import AutoModelForCausalLM
import os

class Generator:
    def __init__(self, model_path, model_type="phi", context_length=4096):
        self.model_path = model_path
        self.model_type = model_type
        self.context_length = context_length
        self.llm = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")     
        print(f"Loading LLM from {self.model_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            model_type=self.model_type,
            context_length=self.context_length,
            gpu_layers=0 
        )

    def generate(self, context, question, max_tokens=512, temperature=0.2):
        prompt = f"""<|user|>
You are a medical assistant. Use the following context to answer the question.
Context:
{context}
Question: {question}
<|end|>
<|assistant|>"""    
        for token in self.llm(
            prompt, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            stop=["<|end|>"],
            stream=True
        ):
            yield token

