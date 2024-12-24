import asyncio
import logging
from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram import flags
from aiogram.fsm.context import FSMContext
from aiogram.types.callback_query import CallbackQuery
from app.states import Gen
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

import app.kb as kb
import app.text as text

import logging
from langchain_milvus import Milvus
from models.models import embedder, llm
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# URI = "./milvus_example_1.db"
URI = "tcp://127.0.0.1:19530"

vector_store = Milvus(
    embedding_function=embedder,
    connection_args={"uri": URI},
)

def format_docs(documents):
    context = []
    for doc in documents:
        url = doc.metadata.get("url", "No URL available")
        title = doc.metadata.get("title", "No Title")
        content = doc.page_content
        context.append(f"Title: {title}\nURL: {url}\nContent: {content}") 
    return "\n\n".join(context)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_bot_and_dispatcher():
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN doesn't exists")
    
    bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    
    return bot, dp

async def generate_text_answer(text_prompt: str) -> str:
    if not llm:
        return "Model is not initialized, generation will not be performed."

    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        Вы являетесь помощником, который находит соответствующие URL-адреса для ответа на вопрос пользователя на основе предоставленных документов.
        Вот контекст с метаданными:
        {context}

        Вопрос: {question}
        Пожалуйста, предоставьте список URL-адресов наиболее релевантных документов.
        """
        )

        qa_chain = (
            {
                "context": vector_store.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return qa_chain.invoke(text_prompt)
    except Exception as e:
        return "Error generating text: " + str(e)


router = Router()

@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(text.greet.format(name=msg.from_user.full_name), reply_markup=kb.menu)

@router.message(F.text == "Меню")
@router.message(F.text == "Выйти в меню")
@router.message(F.text == "◀️ Выйти в меню")
async def menu(msg: Message):
    await msg.answer(text.menu, reply_markup=kb.menu)

@router.callback_query(F.data == "generate_text")
async def input_text_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.text_prompt)
    await clbck.message.edit_text(text.gen_text)
    await clbck.message.answer(text.gen_exit, reply_markup=kb.exit_kb)

@router.message(Gen.text_prompt)
@flags.chat_action("typing")
async def generate_text(msg: Message, state: FSMContext):
    prompt = msg.text
    mesg = await msg.answer(text.gen_wait) 
    res = await generate_text_answer(prompt) 
    if "Error" in res:  
        return await mesg.edit_text(text.gen_error, reply_markup=kb.iexit_kb)
    await mesg.edit_text(res, disable_web_page_preview=True)

async def main():
    logging.basicConfig(level=logging.INFO)
    bot, dp = create_bot_and_dispatcher()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())