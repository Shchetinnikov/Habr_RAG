from chat_builder import get_chat

if __name__ == "__main__":
    model_chat = get_chat()

    model_input = input("Ask LLM anything:")
    print(model_chat.invoke(model_input).content)