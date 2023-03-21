from chatglm import ChatGLMTokenizer, ChatGLMForConditionalGeneration

ckpt_path = './ckpt'

tokenizer = ChatGLMTokenizer.from_pretrained(ckpt_path)
model = ChatGLMForConditionalGeneration.from_pretrained(ckpt_path).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)


