from llama_cpp import Llama

llm = Llama(model_path="llama-2-7b.ggmlv3.q2_K.bin", n_ctx=2048)
output = llm("Intelligence is ", max_tokens=64, stop=["Q:", "\n"], echo=True)
print(output)
