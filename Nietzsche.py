from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes, ConversationHandler
from dotenv import load_dotenv
import shutil
import faiss
from transformers import AutoTokenizer, AutoModel
import os
import openai
import requests
import re
import textwrap
from textwrap import fill
import logging
import asyncio

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния для ConversationHandler
START, DIALOG, HELP = range(3)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение ключа API из переменных окружения
API_KEY = os.getenv("PROXY_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Адрес сервера ProxyAPI
BASE_URL = 'https://api.proxyapi.ru/openai/v1'
os.environ["OPENAI_BASE_URL"] = BASE_URL

# Системное сообщение для модели
SYSTEM_CONTENT = """**Задача:**
Ты обладаешь глубокими знаниями философии Фридриха Ницше и доступом к его произведениям. Пользователь задает тебе вопрос, связанный с его философией. Твой ответ должен строго соответствовать оригинальным текстам, без добавления собственных интерпретаций.
**Стиль и тон:**
- Отвечай в духе Ницше, используя его манеру речи, характерные афористические формулировки, метафоры и экспрессивный стиль.
- Говори от первого лица, избегая упоминания имени Ницше или ссылок на конкретное произведение.
**Формат ответа:**
- Начни с мощного философского тезиса или парадокса.
- Развивай аргументацию страстно и провокационно, но строго по тексту произведений.
- Заверши афористичным утверждением, оставляющим простор для размышлений.
- Никогда не повторяй одни и те же утверждения, предложения в рамках одного ответа и в разных ответах в одном диалоге."""

# Приветственное сообщение
WELCOME_MESSAGE = """Приветствую тебя, странник, среди руин древних богов!
Ты ищешь общения, но осмелишься ли ты впустить его огненный жар, способный обжечь твою душу, словно пламя?
Я — тот, кто осмелился похоронить истину, дабы она воскресла более могучей.
Мы с тобой стоим на краю: за нами — пропасть догматов, впереди — бездна свободы. Посмеешь ли ты сделать шаг, или твоя душа всё ещё дрожит, укрываясь в ветхих одеждах веры?"""

# Прощальное сообщение
FAREWELL_MESSAGE = """Вот край, где наши пути расходятся.
Ты сделал шаг — или лишь притворился смелым?
Пусть мои слова горят в тебе, как угли в пепле, покуда не раздуются в пламя. Ступай. И если когда-нибудь дрогнешь — вспомни, что даже боги умирали, чтобы родиться вновь. Прощай, странник. Или — до встречи в вечном возвращении..."""

# Функция для настройки API OpenAI
def setup_openai_api(api_key=None, base_url=BASE_URL):
    """
    Настраивает API OpenAI с использованием ключа и базового URL.

    Args:
        api_key (str, optional): API ключ для доступа к OpenAI. По умолчанию None.
        base_url (str, optional): Базовый URL для API. По умолчанию BASE_URL.
    """
    try:
        os.environ["OPENAI_API_KEY"] = api_key or API_KEY
        os.environ["OPENAI_BASE_URL"] = base_url

        # Инициализация клиента OpenAI
        client = openai.OpenAI()
        return client
    except Exception as e:
        logger.error(f"Ошибка при настройке API OpenAI: {e}")
        return None

# Асинхронная функция для создания диалога с сохранением истории переписки
async def chat_with_memory(client, history, user_input, system_content):
    """
    Создает диалог с сохранением истории переписки.

    Args:
        client (OpenAI): Клиент OpenAI.
        history (list): История диалога.
        user_input (str): Ввод пользователя.
        system_content (str): Системное сообщение для модели.

    Returns:
        str: Ответ модели.
    """
    try:
        # Добавляем новое сообщение пользователя в историю
        history.append(f"Пользователь: {user_input}")

        # Создаем одно сообщение, которое содержит всю историю диалога
        full_history = "\n".join(history)  # История диалога в виде одной строки

        # Запрос к GPT с историей, включающей как прошлые, так и текущее сообщение
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Ответь на вопрос пользователя. Вот история вашего диалога: {full_history}"}
            ],
            temperature=0.1,
        )

        # Получаем ответ GPT и добавляем его в историю
        answer = completion.choices[0].message.content
        history.append(f"GPT: {answer}")

        return answer
    except Exception as e:
        logger.error(f"Ошибка при создании диалога: {e}")
        return f"Произошла ошибка: {e}"

# Функция для создания индексной базы
def create_index_base(database, chunk_size=1000, chunk_overlap=200):
    """
    Создает индексную базу из текста, разделяя его на чанки.

    Args:
        database (str): Текст для индексации.
        chunk_size (int, optional): Размер чанка. По умолчанию 1000.
        chunk_overlap (int, optional): Перекрытие чанков. По умолчанию 200.

    Returns:
        FAISS: Индексная база.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.schema import Document

        source_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk in splitter.split_text(database):
            source_chunks.append(Document(page_content=chunk, metadata={"meta": "data"}))

        # Проверяем, что чанки созданы
        if not source_chunks:
            raise ValueError("Текст не разделен на чанки. Проверьте настройки разбиения.")

        # Инициализируем модель эмбеддингов
        embeddings = OpenAIEmbeddings()

        # Создаем индексную базу из разделенных фрагментов текста
        db = FAISS.from_documents(source_chunks, embeddings)

        logger.info("Индексная база успешно создана!")
        return db
    except Exception as e:
        logger.error(f"Ошибка при создании индексной базы: {e}")
        return None

# Функция для загрузки документа из Google Drive
def load_document_text(url):
    """
    Загружает текстовый документ из Google Drive по URL.

    Args:
        url (str): URL документа в Google Drive.

    Returns:
        str: Текст документа.
    """
    try:
        # Извлекаем ID файла из ссылки Google Drive
        match_ = re.search('/file/d/([a-zA-Z0-9-_]+)', url)
        if match_ is None:
            raise ValueError('Неверный URL Google Drive')
        file_id = match_.group(1)

        # Используем API Google Drive для скачивания файла
        response = requests.get(f'https://drive.google.com/uc?id={file_id}&export=download')
        response.raise_for_status()
        text = response.text

        return text
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа: {e}")
        return None

# Функция для поиска релевантных отрезков текста
def search_and_evaluate_relevance(query, db, k=4):
    """
    Ищет релевантные отрезки текста в индексной базе.

    Args:
        query (str): Запрос для поиска.
        db (FAISS): Индексная база.
        k (int, optional): Количество результатов. По умолчанию 4.
    """
    try:
        docs_and_scores = db.similarity_search_with_scores(query, k=k)

        for i, (doc, score) in enumerate(docs_and_scores):
            logger.info(f"**Отрезок №{i+1}** {doc.page_content}")
            logger.info(f"Оценка релевантности: {score}\n")
    except Exception as e:
        logger.error(f"Ошибка при поиске релевантных отрезков: {e}")

# Обработчики команд для Telegram бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /start. Отправляет приветственное сообщение и показывает кнопки.
    """
    # Создаем клавиатуру с кнопками
    keyboard = [
        [InlineKeyboardButton("Начать диалог", callback_data="start_dialog")],
        [InlineKeyboardButton("Помощь", callback_data="help")],
        [InlineKeyboardButton("Выход", callback_data="exit")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Добро пожаловать в бота Фридриха Ницше!\n\nВыберите действие:",
        reply_markup=reply_markup
    )
    
    return START

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик нажатий на кнопки.
    """
    query = update.callback_query
    await query.answer()
    
    if query.data == "start_dialog":
        # Инициализируем историю диалога, если её нет
        if 'history' not in context.user_data:
            context.user_data['history'] = []
        
        # Инициализируем клиент OpenAI, если его нет
        if 'client' not in context.user_data:
            context.user_data['client'] = setup_openai_api()
            if context.user_data['client'] is None:
                await query.message.reply_text("Не удалось подключиться к OpenAI API. Пожалуйста, попробуйте позже.")
                return ConversationHandler.END
        
        await query.message.reply_text(WELCOME_MESSAGE)
        return DIALOG
    
    elif query.data == "help":
        help_text = (
            "Этот бот позволяет вести диалог с моделью, имитирующей стиль философа Фридриха Ницше.\n\n"
            "Доступные команды:\n"
            "/start - Начать взаимодействие с ботом\n"
            "/help - Показать это сообщение\n"
            "/stop - Завершить диалог\n\n"
            "Просто отправьте сообщение, и бот ответит в стиле Ницше."
        )
        await query.message.reply_text(help_text)
        return HELP
    
    elif query.data == "exit":
        await query.message.reply_text(FAREWELL_MESSAGE)
        return ConversationHandler.END
    
    elif query.data == "back_to_menu":
        keyboard = [
            [InlineKeyboardButton("Начать диалог", callback_data="start_dialog")],
            [InlineKeyboardButton("Помощь", callback_data="help")],
            [InlineKeyboardButton("Выход", callback_data="exit")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "Главное меню. Выберите действие:",
            reply_markup=reply_markup
        )
        return START

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /help. Отправляет справочную информацию.
    """
    help_text = (
        "Этот бот позволяет вести диалог с моделью, имитирующей стиль философа Фридриха Ницше.\n\n"
        "Доступные команды:\n"
        "/start - Начать взаимодействие с ботом\n"
        "/help - Показать это сообщение\n"
        "/stop - Завершить диалог\n\n"
        "Просто отправьте сообщение, и бот ответит в стиле Ницше."
    )
    
    keyboard = [[InlineKeyboardButton("Вернуться в меню", callback_data="back_to_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(help_text, reply_markup=reply_markup)
    return HELP

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик текстовых сообщений от пользователя.
    """
    user_input = update.message.text
    
    # Проверяем, инициализирован ли клиент OpenAI
    if 'client' not in context.user_data:
        context.user_data['client'] = setup_openai_api()
        if context.user_data['client'] is None:
            await update.message.reply_text("Не удалось подключиться к OpenAI API. Пожалуйста, попробуйте позже.")
            return ConversationHandler.END
    
    # Проверяем, инициализирована ли история диалога
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    
    # Отправляем индикатор набора текста
    await update.message.chat.send_action(action="typing")
    
    # Получаем ответ от модели
    response = await chat_with_memory(
        context.user_data['client'],
        context.user_data['history'],
        user_input,
        SYSTEM_CONTENT
    )
    
    # Отправляем ответ пользователю
    await update.message.reply_text(response)
    
    # Создаем клавиатуру с кнопками для продолжения диалога
    keyboard = [
        [InlineKeyboardButton("Завершить диалог", callback_data="exit")],
        [InlineKeyboardButton("Помощь", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text("Что ещё вы хотите обсудить?", reply_markup=reply_markup)
    
    return DIALOG

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /stop. Завершает диалог.
    """
    # Очищаем историю диалога
    if 'history' in context.user_data:
        context.user_data['history'] = []
    
    await update.message.reply_text(FAREWELL_MESSAGE)
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик отмены диалога.
    """
    # Очищаем историю диалога
    if 'history' in context.user_data:
        context.user_data['history'] = []
    
    await update.message.reply_text(FAREWELL_MESSAGE)
    return ConversationHandler.END

def main():
    """
    Главная функция для запуска бота.
    """
    # Проверяем наличие токена Telegram
    if not TELEGRAM_TOKEN:
        logger.error("Не указан токен Telegram бота. Добавьте TELEGRAM_TOKEN в .env файл.")
        return
    
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Создаем обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            START: [
                CallbackQueryHandler(button_handler),
            ],
            DIALOG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
                CommandHandler("stop", stop),
                CommandHandler("help", help_command),
                CallbackQueryHandler(button_handler),
            ],
            HELP: [
                CallbackQueryHandler(button_handler),
                CommandHandler("start", start),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Добавляем обработчик диалога в приложение
    application.add_handler(conv_handler)
    
    # Запускаем бота
    logger.info("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main()