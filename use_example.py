# pip install hiercontext 

from hiercontext import make_context

context_root = "accelerate/utils"
context = make_context(path=context_root, model="Qwen")


